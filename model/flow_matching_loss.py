import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x, dim=-1, eps=1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


class FlowMatchingLoss(nn.Module):
    def __init__(
        self,
        flow_net,
        num_steps=4,
        lambda_fm=1.0,
        lambda_end=1.0,
        lambda_ret=0.05,
        normalize=True,
        temperature=0.07,
        gamma=1.5,
        eps=1e-6,
        path_type="linear",
        geodesic_eps=1e-4,
        step_normalize=True,
    ):
        """
        A simplified and more stable flow matching loss.

        Args
        ----
        flow_net : velocity network
        num_steps : Euler integration steps
        lambda_fm : weight for flow matching loss
        lambda_end : weight for endpoint loss
        normalize : whether to l2 normalize embeddings
        gamma : nonlinear time schedule alpha(t)=t^gamma
        eps : numerical stability
        """
        super().__init__()

        self.flow_net = flow_net
        self.num_steps = num_steps
        self.lambda_fm = lambda_fm
        self.lambda_end = lambda_end
        self.lambda_ret = lambda_ret
        self.normalize = normalize
        self.temperature = temperature
        self.gamma = gamma
        self.eps = eps
        self.path_type = path_type
        self.geodesic_eps = geodesic_eps
        self.step_normalize = step_normalize

    def _maybe_normalize_inputs(self, q, y, e_m=None):
        if self.normalize:
            q = l2norm(q, eps=self.eps)
            y = l2norm(y, eps=self.eps)
            if e_m is not None:
                e_m = l2norm(e_m, eps=self.eps)
        return q, y, e_m

    def _linear_path_target(self, q, y, t):
        """Euclidean / linear path with constant velocity."""
        x_t = (1.0 - t) * q + t * y
        u_star = y - q
        return x_t, u_star

    def _geodesic_path_target(self, q, y, t):
        """
        Geodesic path on unit sphere (slerp) and its analytic velocity:
            gamma(t) = sin((1-t)theta)/sin(theta) * q + sin(t*theta)/sin(theta) * y
            theta = arccos(q^T y)
        """
        q_n = l2norm(q, eps=self.eps)
        y_n = l2norm(y, eps=self.eps)

        dot = (q_n * y_n).sum(dim=-1, keepdim=True).clamp(-1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)

        near_singular = sin_theta.abs() < self.geodesic_eps
        sin_theta_safe = sin_theta.clamp(min=self.geodesic_eps)

        coeff_q = torch.sin((1.0 - t) * theta) / sin_theta_safe
        coeff_y = torch.sin(t * theta) / sin_theta_safe
        x_t_geo = coeff_q * q_n + coeff_y * y_n

        vel_q = -theta * torch.cos((1.0 - t) * theta) / sin_theta_safe
        vel_y = theta * torch.cos(t * theta) / sin_theta_safe
        u_star_geo = vel_q * q_n + vel_y * y_n

        x_t_lin, u_star_lin = self._linear_path_target(q_n, y_n, t)
        x_t = torch.where(near_singular, x_t_lin, x_t_geo)
        u_star = torch.where(near_singular, u_star_lin, u_star_geo)
        return x_t, u_star

    def _path_target(self, q, y, t):
        if self.path_type == "linear":
            return self._linear_path_target(q, y, t)
        if self.path_type == "geodesic":
            return self._geodesic_path_target(q, y, t)
        raise ValueError(f"Unsupported path_type: {self.path_type}")

    def _flow_net_call(self, x, q0, e_m, t):
        """
        Reduce shortcut:
        pass delta = x - q0 instead of raw q0.
        """
        delta = x - q0
        v = self.flow_net(x, delta=delta, e_m=e_m, t=t)
        v = torch.tanh(v)
        return v

    def integrate_flow(self, q, e_m=None):
        """
        Simple Euler integration with per-step normalization.
        """
        x = q.clone()
        B = q.size(0)
        device = q.device
        dt = 1.0 / self.num_steps

        for k in range(self.num_steps):
            t = torch.full(
                (B, 1),
                k / self.num_steps,
                device=device,
                dtype=q.dtype,
            )

            v = self._flow_net_call(x, q, e_m, t)
            x = x + dt * v

            if self.step_normalize:
                x = l2norm(x, eps=self.eps)

        return x

    def forward(self, q, y, e_m=None):
        """
        q   : source/query feature [B, D]
        y   : target/modified feature [B, D]
        e_m : optional condition feature [B, D]
        """
        q, y, e_m = self._maybe_normalize_inputs(q, y, e_m)

        B = q.size(0)
        device = q.device
        dtype = q.dtype

        # 1) flow matching loss
        t = torch.rand(B, 1, device=device, dtype=dtype)

        x_t, u_star = self._path_target(q, y, t)
        u_pred = self._flow_net_call(x_t, q, e_m, t)

        loss_fm = F.mse_loss(u_pred, u_star)

        # 2) endpoint loss
        y_hat = self.integrate_flow(q, e_m)
        loss_end = F.mse_loss(y_hat, y)
        # mse_end = F.mse_loss(y_hat, y)
        # cos_end = (1.0 - F.cosine_similarity(y_hat, y, dim=-1)).mean()
        
        # loss_end = 0.5 * mse_end + 0.5 * cos_end

        # 3) Retrieval Loss (optional)
        loss_ret = torch.tensor(0.0, device=device, dtype=dtype)

        if self.lambda_ret > 0:
            y_hat_n = l2norm(y_hat, eps=self.eps) if not self.normalize else y_hat
            y_n = l2norm(y, eps=self.eps) if not self.normalize else y

            logits = y_hat_n @ y_n.t() / self.temperature
            labels = torch.arange(B, device=device)

            loss_ret = F.cross_entropy(logits, labels)

        # 4) total loss
        loss = self.lambda_fm * loss_fm + self.lambda_end * loss_end + self.lambda_ret * loss_ret

        return {
            "loss": loss,
            "loss_fm": loss_fm,
            "loss_end": loss_end,
            "loss_ret": loss_ret,
            "y_hat": y_hat,
        }

# v3
# def l2norm(x, dim=-1, eps=1e-6):
#     return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


# class FlowMatchingLoss(nn.Module):
#     def __init__(
#         self,
#         flow_net,
#         num_steps=8,
#         lambda_fm=1.0,
#         lambda_mid=0.5,
#         lambda_end=1.0,
#         lambda_ret=0.0,
#         temperature=0.07,
#         normalize=True,
#         gamma=2.0,
#         eps=1e-6,
#     ):
#         """
#         Geometry-aware flow matching for CLIP-like global tokens.

#         Args
#         ----
#         flow_net : velocity network v_theta
#         num_steps : sphere-Euler integration steps
#         lambda_fm : weight for flow matching loss
#         lambda_mid : weight for random intermediate-state consistency
#         lambda_end : weight for endpoint loss
#         lambda_ret : weight for retrieval contrastive loss
#         temperature : contrastive temperature
#         normalize : whether to l2 normalize inputs / states
#         gamma : power for nonlinear time schedule alpha(t)=t^gamma
#         eps : numerical stability
#         """
#         super().__init__()

#         self.flow_net = flow_net
#         self.num_steps = num_steps

#         self.lambda_fm = lambda_fm
#         self.lambda_mid = lambda_mid
#         self.lambda_end = lambda_end
#         self.lambda_ret = lambda_ret

#         self.temperature = temperature
#         self.normalize = normalize
#         self.gamma = gamma
#         self.eps = eps

#     def _maybe_normalize_inputs(self, q, y, e_m):
#         if self.normalize:
#             q = l2norm(q, eps=self.eps)
#             y = l2norm(y, eps=self.eps)
#             e_m = l2norm(e_m, eps=self.eps)
#         return q, y, e_m

#     def _alpha(self, t):
#         return t.pow(self.gamma)

#     def _project_to_tangent(self, x, v):
#         """
#         Project v onto the tangent space at x on the unit sphere.
#         x, v: [B, D]
#         """
#         return v - (x * v).sum(dim=-1, keepdim=True) * x

#     def _slerp(self, x0, x1, t):
#         """
#         Spherical linear interpolation between x0 and x1.
#         x0, x1: [B, D], assumed normalized
#         t: [B, 1] in [0, 1]
#         """
#         x0 = l2norm(x0, eps=self.eps)
#         x1 = l2norm(x1, eps=self.eps)

#         dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1.0 + self.eps, 1.0 - self.eps)
#         theta = torch.acos(dot)                              # [B, 1]
#         sin_theta = torch.sin(theta).clamp(min=self.eps)    # [B, 1]

#         w0 = torch.sin((1.0 - t) * theta) / sin_theta
#         w1 = torch.sin(t * theta) / sin_theta

#         x_t = w0 * x0 + w1 * x1
#         return l2norm(x_t, eps=self.eps)

#     def _path_target(self, q, y, t):
#         """
#         Geometry-aware path on the unit sphere:
#             x_t = slerp(q, y, alpha(t))

#         State-dependent target velocity in tangent space:
#             u_star = Proj_{T_{x_t}}(y - x_t)

#         This avoids the constant-velocity shortcut.
#         """
#         alpha = self._alpha(t)
#         x_t = self._slerp(q, y, alpha)
#         u_star = self._project_to_tangent(x_t, y - x_t)
#         return x_t, u_star

#     def _flow_net_call(self, x, q0, e_m, t):
#         """
#         Shortcut-avoidance trick:
#         feed relative state delta=x-q0 instead of raw q0.
#         Keep interface compatible with flow_net(x_t, q_like, e_m, t).
#         """
#         delta = x - q0
#         v = self.flow_net(x, delta, e_m, t)
#         v = self._project_to_tangent(x, v)
#         return v

#     def _sphere_euler_step(self, x, v, dt):
#         """
#         Tangent-space Euler step + renormalization.
#         """
#         v = self._project_to_tangent(x, v)
#         x = x + dt * v
#         if self.normalize:
#             x = l2norm(x, eps=self.eps)
#         return x

#     def integrate_flow(self, q, e_m, return_mid=False, mid_step=None):
#         """
#         Integrate velocity field from source feature q on the sphere.
#         """
#         x = q.clone()
#         B = q.size(0)
#         device = q.device
#         dt = 1.0 / self.num_steps

#         x_mid = None

#         for k in range(self.num_steps):
#             t = torch.full(
#                 (B, 1),
#                 k / self.num_steps,
#                 device=device,
#                 dtype=q.dtype,
#             )

#             v = self._flow_net_call(x, q, e_m, t)
#             x = self._sphere_euler_step(x, v, dt)

#             if return_mid and mid_step is not None and (k + 1) == mid_step:
#                 x_mid = x.clone()

#         if return_mid:
#             return x, x_mid
#         return x

#     def forward(self, q, y, e_m):
#         """
#         q   : source/query feature [B, D]
#         y   : target/modified feature [B, D]
#         e_m : modification text feature [B, D]
#         """
#         q, y, e_m = self._maybe_normalize_inputs(q, y, e_m)

#         B = q.size(0)
#         device = q.device
#         dtype = q.dtype

#         # --------------------------------------------------
#         # 1) Geometry-aware Flow Matching Loss
#         # --------------------------------------------------
#         t = torch.rand(B, 1, device=device, dtype=dtype)

#         x_t, u_star = self._path_target(q, y, t)
#         u_pred = self._flow_net_call(x_t, q, e_m, t)

#         loss_fm = F.mse_loss(u_pred, u_star)

#         # --------------------------------------------------
#         # 2) Intermediate Consistency Loss
#         # --------------------------------------------------
#         loss_mid = torch.tensor(0.0, device=device, dtype=dtype)

#         if self.lambda_mid > 0 and self.num_steps >= 2:
#             mid_step = torch.randint(
#                 low=1,
#                 high=self.num_steps,
#                 size=(1,),
#                 device=device,
#             ).item()

#             _, x_mid = self.integrate_flow(q, e_m, return_mid=True, mid_step=mid_step)

#             t_mid = torch.full(
#                 (B, 1),
#                 mid_step / self.num_steps,
#                 device=device,
#                 dtype=dtype,
#             )

#             x_mid_target, _ = self._path_target(q, y, t_mid)
#             loss_mid = F.mse_loss(x_mid, x_mid_target)

#         # --------------------------------------------------
#         # 3) Endpoint Loss
#         # --------------------------------------------------
#         y_hat = self.integrate_flow(q, e_m)
#         loss_end = F.mse_loss(y_hat, y)

#         # --------------------------------------------------
#         # 4) Retrieval Loss (optional)
#         # --------------------------------------------------
#         loss_ret = torch.tensor(0.0, device=device, dtype=dtype)

#         if self.lambda_ret > 0:
#             y_hat_n = l2norm(y_hat, eps=self.eps) if not self.normalize else y_hat
#             y_n = l2norm(y, eps=self.eps) if not self.normalize else y

#             logits = y_hat_n @ y_n.t() / self.temperature
#             labels = torch.arange(B, device=device)

#             loss_ret = F.cross_entropy(logits, labels)

#         # --------------------------------------------------
#         # 5) Total Loss
#         # --------------------------------------------------
#         loss = (
#             self.lambda_fm * loss_fm
#             + self.lambda_mid * loss_mid
#             + self.lambda_end * loss_end
#             + self.lambda_ret * loss_ret
#         )

#         return {
#             "loss": loss,
#             "loss_fm": loss_fm,
#             "loss_mid": loss_mid,
#             "loss_end": loss_end,
#             "loss_ret": loss_ret,
#             "y_hat": y_hat,
#         }
# v2
# def l2norm(x, dim=-1, eps=1e-6):
#     return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


# class FlowMatchingLoss(nn.Module):
#     def __init__(
#         self,
#         flow_net,
#         num_steps=8,
#         lambda_fm=1.0,
#         lambda_mid=0.5,
#         lambda_end=1.0,
#         lambda_ret=0.0,
#         temperature=0.07,
#         normalize=True,
#         gamma=2.0,
#     ):
#         """
#         Args
#         ----
#         flow_net : velocity network v_theta
#         num_steps : Euler integration steps
#         lambda_fm : weight for flow matching loss
#         lambda_mid : weight for random intermediate-state consistency
#         lambda_end : weight for endpoint loss
#         lambda_ret : weight for retrieval contrastive loss
#         temperature : contrastive temperature
#         normalize : whether to l2 normalize embeddings
#         gamma : power for nonlinear path alpha(t)=t^gamma
#         """
#         super().__init__()

#         self.flow_net = flow_net
#         self.num_steps = num_steps

#         self.lambda_fm = lambda_fm
#         self.lambda_mid = lambda_mid
#         self.lambda_end = lambda_end
#         self.lambda_ret = lambda_ret

#         self.temperature = temperature
#         self.normalize = normalize
#         self.gamma = gamma

#     def _maybe_normalize_inputs(self, q, y, e_m):
#         if self.normalize:
#             q = l2norm(q)
#             y = l2norm(y)
#             e_m = l2norm(e_m)
#         return q, y, e_m

#     def _alpha(self, t):
#         # nonlinear schedule: alpha(t)=t^gamma
#         return t.pow(self.gamma)

#     def _path_target(self, q, y, t):
#         """
#         Nonlinear path from q to y:
#             x_t = (1-alpha) q + alpha y
#         but supervision uses state-dependent residual:
#             u_star = y - x_t

#         This is intentionally NOT the exact derivative of x_t,
#         because the goal here is to avoid the shortcut of learning
#         a sample-wise constant velocity.
#         """
#         alpha = self._alpha(t)
#         x_t = (1.0 - alpha) * q + alpha * y
#         u_star = y - x_t
#         return x_t, u_star

#     def _flow_net_call(self, x, q0, e_m, t):
#         """
#         Shortcut-avoidance trick:
#         do NOT feed raw q0 directly.
#         Instead feed relative state delta = x - q0 as the second argument.
#         This keeps the existing flow_net(x_t, q, e_m, t) interface unchanged,
#         while weakening the direct q -> target shortcut.
#         """
#         delta = x - q0
#         return self.flow_net(x, delta, e_m, t)

#     def integrate_flow(self, q, e_m, return_mid=False, mid_step=None):
#         """
#         Integrate velocity field from source feature q.
#         """
#         x = q.clone()
#         B = q.size(0)
#         device = q.device
#         dt = 1.0 / self.num_steps

#         x_mid = None

#         for k in range(self.num_steps):
#             t = torch.full(
#                 (B, 1),
#                 k / self.num_steps,
#                 device=device,
#                 dtype=q.dtype,
#             )

#             v = self._flow_net_call(x, q, e_m, t)
#             x = x + dt * v

#             if return_mid and mid_step is not None and (k + 1) == mid_step:
#                 x_mid = x.clone()

#         if self.normalize:
#             x = l2norm(x)
#             if x_mid is not None:
#                 x_mid = l2norm(x_mid)

#         if return_mid:
#             return x, x_mid
#         return x

#     def forward(self, q, y, e_m):
#         """
#         q   : source/query feature [B, D]
#         y   : target/modified feature [B, D]
#         e_m : modification text feature [B, D]
#         """
#         q, y, e_m = self._maybe_normalize_inputs(q, y, e_m)

#         B = q.size(0)
#         device = q.device

#         # --------------------------------------------------
#         # 1 Flow Matching Loss
#         # --------------------------------------------------
#         t = torch.rand(B, 1, device=device, dtype=q.dtype)

#         # nonlinear path + state-dependent target
#         x_t, u_star = self._path_target(q, y, t)

#         # do not pass raw q directly; pass relative state through wrapper
#         u_pred = self._flow_net_call(x_t, q, e_m, t)

#         loss_fm = F.mse_loss(u_pred, u_star)

#         # --------------------------------------------------
#         # 2 Intermediate Consistency Loss
#         # --------------------------------------------------
#         loss_mid = torch.tensor(0.0, device=device, dtype=q.dtype)

#         if self.lambda_mid > 0 and self.num_steps >= 2:
#             mid_step = torch.randint(
#                 low=1,
#                 high=self.num_steps,
#                 size=(1,),
#                 device=device,
#             ).item()

#             _, x_mid = self.integrate_flow(q, e_m, return_mid=True, mid_step=mid_step)

#             t_mid = torch.full(
#                 (B, 1),
#                 mid_step / self.num_steps,
#                 device=device,
#                 dtype=q.dtype,
#             )
#             x_mid_target, _ = self._path_target(q, y, t_mid)

#             if self.normalize:
#                 x_mid_target = l2norm(x_mid_target)

#             loss_mid = F.mse_loss(x_mid, x_mid_target)

#         # --------------------------------------------------
#         # 3 Endpoint Loss
#         # --------------------------------------------------
#         y_hat = self.integrate_flow(q, e_m)

#         loss_end = F.mse_loss(y_hat, y)

#         # --------------------------------------------------
#         # 4 Retrieval Loss (optional)
#         # --------------------------------------------------
#         loss_ret = torch.tensor(0.0, device=device, dtype=q.dtype)

#         if self.lambda_ret > 0:
#             y_hat_n = l2norm(y_hat) if not self.normalize else y_hat
#             y_n = l2norm(y) if not self.normalize else y

#             logits = y_hat_n @ y_n.t() / self.temperature
#             labels = torch.arange(B, device=device)

#             loss_ret = F.cross_entropy(logits, labels)

#         # --------------------------------------------------
#         # 5 Total Loss
#         # --------------------------------------------------
#         loss = (
#             self.lambda_fm * loss_fm
#             + self.lambda_mid * loss_mid
#             + self.lambda_end * loss_end
#             + self.lambda_ret * loss_ret
#         )

#         return {
#             "loss": loss,
#             "loss_fm": loss_fm,
#             "loss_mid": loss_mid,
#             "loss_end": loss_end,
#             "loss_ret": loss_ret,
#             "y_hat": y_hat,
#         }

# v1
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def l2norm(x, dim=-1, eps=1e-6):
#     return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


# class FlowMatchingLoss(nn.Module):

#     def __init__(
#         self,
#         flow_net,
#         num_steps=4,
#         lambda_fm=1.0,
#         lambda_end=1.0,
#         lambda_ret=0.0,
#         temperature=0.07,
#         normalize=True,
#     ):
#         """
#         Args
#         ----
#         flow_net : velocity network v_theta
#         num_steps : integration steps
#         lambda_fm : weight for flow matching loss
#         lambda_end : weight for endpoint loss
#         lambda_ret : weight for retrieval contrastive loss
#         temperature : contrastive temperature
#         normalize : whether to l2 normalize embeddings
#         """

#         super().__init__()

#         self.flow_net = flow_net
#         self.num_steps = num_steps

#         self.lambda_fm = lambda_fm
#         self.lambda_end = lambda_end
#         self.lambda_ret = lambda_ret

#         self.temperature = temperature
#         self.normalize = normalize

#     def integrate_flow(self, q, e_m):
#         """
#         Integrate velocity field from source feature q
#         """

#         x = q.clone()
#         B = q.size(0)
#         device = q.device
#         dt = 1.0 / self.num_steps

#         for k in range(self.num_steps):

#             t = torch.full(
#                 (B, 1),
#                 k / self.num_steps,
#                 device=device,
#                 dtype=q.dtype,
#             )

#             v = self.flow_net(x, q, e_m, t)
#             x = x + dt * v

#         if self.normalize:
#             x = l2norm(x)

#         return x

#     def forward(self, q, y, e_m):
#         """
#         q : composed query feature [B, D]
#         y : modified description feature [B, D]
#         e_m : modification text feature [B, D]
#         """

#         B = q.size(0)
#         device = q.device

#         # --------------------------------------------------
#         # 1 Flow Matching Loss
#         # --------------------------------------------------

#         t = torch.rand(B, 1, device=device, dtype=q.dtype)

#         x_t = (1.0 - t) * q + t * y
#         u_star = y - q

#         u_pred = self.flow_net(x_t, q, e_m, t)

#         loss_fm = F.mse_loss(u_pred, u_star)

#         # --------------------------------------------------
#         # 2 Endpoint Loss
#         # --------------------------------------------------

#         y_hat = self.integrate_flow(q, e_m)

#         loss_end = F.mse_loss(y_hat, y)

#         # --------------------------------------------------
#         # 3 Retrieval Loss (optional)
#         # --------------------------------------------------

#         loss_ret = torch.tensor(0.0, device=device)

#         if self.lambda_ret > 0:

#             if self.normalize:
#                 y_hat = l2norm(y_hat)
#                 y = l2norm(y)

#             logits = y_hat @ y.t() / self.temperature
#             labels = torch.arange(B, device=device)

#             loss_ret = F.cross_entropy(logits, labels)

#         # --------------------------------------------------
#         # 4 Total Loss
#         # --------------------------------------------------

#         loss = (
#             self.lambda_fm * loss_fm
#             + self.lambda_end * loss_end
#             + self.lambda_ret * loss_ret
#         )

#         return {
#             "loss": loss,
#             "loss_fm": loss_fm,
#             "loss_end": loss_end,
#             "loss_ret": loss_ret,
#             "y_hat": y_hat,
#         }
