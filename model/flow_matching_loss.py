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


def sphere_expmap_step(x, v, dt, eps=1e-6):
    """
    Exponential-map update on unit sphere:
        x_next = cos(dt*||v_tan||) * x + sin(dt*||v_tan||) * v_tan/||v_tan||
    where v_tan is the projection of v onto the tangent space at x.
    """
    x_unit = l2norm(x, eps=eps)
    v_tan = v - (v * x_unit).sum(dim=-1, keepdim=True) * x_unit
    speed = v_tan.norm(dim=-1, keepdim=True).clamp(min=eps)
    theta = dt * speed
    direction = v_tan / speed
    x_next = torch.cos(theta) * x_unit + torch.sin(theta) * direction
    return l2norm(x_next, eps=eps)


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
        step_norm_type="l2",
        hybrid_geodesic_steps=0,
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
        self.step_norm_type = step_norm_type
        self.hybrid_geodesic_steps = max(0, int(hybrid_geodesic_steps))

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
        if self.hybrid_geodesic_steps > 0 and self.num_steps > 0:
            return self._hybrid_path_target(q, y, t)
        if self.path_type == "linear":
            return self._linear_path_target(q, y, t)
        if self.path_type == "geodesic":
            return self._geodesic_path_target(q, y, t)
        raise ValueError(f"Unsupported path_type: {self.path_type}")

    def _hybrid_path_target(self, q, y, t):
        """
        Piecewise target path for hybrid mode:
          - first tau=s/N: geodesic target
          - last 1-tau   : linear target towards y
        """
        tau = float(self.hybrid_geodesic_steps) / float(self.num_steps)
        if tau <= 0.0:
            return self._linear_path_target(q, y, t)
        if tau >= 1.0:
            return self._geodesic_path_target(q, y, t)

        x_geo_t, u_geo_t = self._geodesic_path_target(q, y, t)
        tau_tensor = torch.full_like(t, tau)
        x_tau, _ = self._geodesic_path_target(q, y, tau_tensor)

        remain = max(1.0 - tau, self.eps)
        alpha = (t - tau_tensor) / remain
        alpha = alpha.clamp(0.0, 1.0)
        x_lin_t = (1.0 - alpha) * x_tau + alpha * y
        u_lin_t = (y - x_tau) / remain

        late_mask = t >= tau_tensor
        x_t = torch.where(late_mask, x_lin_t, x_geo_t)
        u_star = torch.where(late_mask, u_lin_t, u_geo_t)
        return x_t, u_star

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
        hybrid_geodesic_steps = max(0, min(self.hybrid_geodesic_steps, int(self.num_steps)))

        for k in range(self.num_steps):
            t = torch.full(
                (B, 1),
                k / self.num_steps,
                device=device,
                dtype=q.dtype,
            )

            v = self._flow_net_call(x, q, e_m, t)
            if hybrid_geodesic_steps > 0:
                if k < hybrid_geodesic_steps:
                    if self.step_norm_type == "expmap":
                        x = sphere_expmap_step(x, v, dt=dt, eps=self.eps)
                    else:
                        x = l2norm(x + dt * v, eps=self.eps)
                else:
                    x = x + dt * v
            else:
                if self.step_normalize:
                    if self.step_norm_type == "expmap":
                        x = sphere_expmap_step(x, v, dt=dt, eps=self.eps)
                    else:
                        x = l2norm(x + dt * v, eps=self.eps)
                else:
                    x = x + dt * v

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
        
        # loss_end = cos_end

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
