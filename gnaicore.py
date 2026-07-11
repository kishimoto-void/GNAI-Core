import jax
import jax.numpy as jnp
from jax import grad, value_and_grad
from functools import partial
from dataclasses import dataclass, replace
from jax.tree_util import register_pytree_node
import jax.random as jr


# =====================================================================
# 1. CONFIG / STATE / PARAMS（最小化）
# =====================================================================
@dataclass
class Config:
    n: int = 4
    d: int = 3
    dt: float = 0.01
    lr_s: float = 0.50
    lr_p: float = 0.02

    pi_prop: float = 4.0
    λ_exp: float = 1.2
    α_res: float = 0.4
    λ_filt: float = 2.0
    λ_trans: float = 0.6
    λ_str: float = 0.3
    λ_θ: float = 0.01
    w_reg: float = 0.01
    huber: float = 0.8

    @property
    def h(self) -> int:
        return max(16, self.n * self.d * 2)


@dataclass
class State:
    x: jnp.ndarray        # (n, d)
    mu: jnp.ndarray       # (n, d)
    mu_prev: jnp.ndarray  # (n, d)
    r: jnp.ndarray        # (n, d)


@dataclass
class Params:
    # Graph & Attention
    W_geo: jnp.ndarray    # (n, n)
    W_query: jnp.ndarray  # (d, d)
    W_key: jnp.ndarray    # (d, d)
    
    # Shared Encoder + Heads
    W_enc1: jnp.ndarray; b_enc1: jnp.ndarray
    W_enc2: jnp.ndarray; b_enc2: jnp.ndarray
    W_pred: jnp.ndarray; b_pred: jnp.ndarray   # prediction head
    W_trn: jnp.ndarray; b_trn: jnp.ndarray     # transition head
    W_prec: jnp.ndarray; b_prec: jnp.ndarray   # precision head


def register_pytree(cls):
    fields = [f for f in cls.__dataclass_fields__]
    register_pytree_node(
        cls,
        lambda o: ([getattr(o, f) for f in fields], None),
        lambda _, tup: cls(**dict(zip(fields, tup)))
    )

register_pytree(State)
register_pytree(Params)


# =====================================================================
# 2. GNAI CORE（最小クラスに集約）
# =====================================================================
class GNAICore:
    def __init__(self, cfg: Config, seed: int = 42):
        self.cfg = cfg
        self.key = jr.PRNGKey(seed)
        self.kernel_fn = self._mexican_hat(0.6)

    @staticmethod
    def _mexican_hat(σ_ph: float):
        return lambda dist: (1.0 - 2.0 * (dist**2 / (2 * σ_ph**2))) * jnp.exp(-dist**2 / (2 * σ_ph**2))

    def init_params(self) -> Params:
        k = jr.split(self.key, 10)
        h = self.cfg.h
        n, d = self.cfg.n, self.cfg.d

        return Params(
            W_geo=jr.uniform(k[0], (n, n), minval=-0.1, maxval=0.1),
            W_query=jr.uniform(k[1], (d, d), minval=-0.1, maxval=0.1),
            W_key=jr.uniform(k[2], (d, d), minval=-0.1, maxval=0.1),
            W_enc1=jr.uniform(k[3], (2*d, h), minval=-0.1, maxval=0.1),
            b_enc1=jnp.zeros(h),
            W_enc2=jr.uniform(k[4], (h, h), minval=-0.1, maxval=0.1),
            b_enc2=jnp.zeros(h),
            W_pred=jr.uniform(k[5], (h, d), minval=-0.1, maxval=0.1),
            b_pred=jnp.zeros(d),
            W_trn=jr.uniform(k[6], (h, d), minval=-0.1, maxval=0.1),
            b_trn=jnp.zeros(d),
            W_prec=jr.uniform(k[7], (h, d), minval=-0.1, maxval=0.1),
            b_prec=jnp.zeros(d),
        )

    def init_state(self) -> State:
        k = jr.split(self.key, 3)
        n, d = self.cfg.n, self.cfg.d
        x0 = jr.uniform(k[0], (n, d), minval=-0.5, maxval=0.5)
        return State(x=x0, mu=x0, mu_prev=x0, r=jnp.zeros((n, d)))

    # ------------------- Free Energy -------------------
    @partial(jax.jit, static_argnums=(0,))
    def _free_energy(self, s: State, p: Params, target: jnp.ndarray):
        n, d = self.cfg.n, self.cfg.d
        A = jax.nn.softplus((p.W_geo + p.W_geo.T) / 2.0) * (1.0 - jnp.eye(n))
        D = jnp.diag(jnp.sum(A, axis=1))
        L = D - A

        # Graph Attention
        q = s.mu @ p.W_query
        k = s.mu @ p.W_key
        attn = jax.nn.softmax(jnp.where(A > 0, (q @ k.T) / jnp.sqrt(d), -1e9), axis=-1)
        msg = attn @ s.mu

        # Shared Encoder
        feat = jnp.concatenate([s.mu, msg], axis=-1)
        h = jax.nn.silu(feat @ p.W_enc1 + p.b_enc1)
        h = jax.nn.silu(h @ p.W_enc2 + p.b_enc2)

        # Heads
        y_hat = h @ p.W_pred + p.b_pred
        f_mu = h @ p.W_trn + p.b_trn
        pi = jax.nn.softplus(h @ p.W_prec + p.b_prec)

        # Energy Terms
        e_ex = s.x - y_hat
        huber = jnp.where(jnp.abs(e_ex) <= self.cfg.huber,
                         0.5 * e_ex**2,
                         self.cfg.huber * (jnp.abs(e_ex) - 0.5 * self.cfg.huber))

        F_err = jnp.sum(pi * huber - jnp.log(pi + 1e-12) + 0.5 * self.cfg.pi_prop * (s.x - s.mu)**2)
        F_exp = 0.5 * self.cfg.λ_exp * jnp.sum((s.mu - ((1-self.cfg.α_res)*target + self.cfg.α_res*s.r))**2) \
              + 0.5 * self.cfg.λ_filt * jnp.sum((s.r - e_ex)**2)

        mu_target = s.mu_prev + self.cfg.dt * f_mu
        F_trn = 0.5 * self.cfg.λ_trans * jnp.sum((s.mu - mu_target)**2)

        # Geometry
        diff = s.x[:, None, :] - s.x[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)
        F_geo = 0.25 * (jnp.sum(s.mu * (L @ s.mu)) +
                        jnp.sum(A * (self.cfg.λ_str * dist**2 - self.kernel_fn(dist))))

        F_reg = 0.5 * self.cfg.λ_θ * sum(jnp.sum(w**2) for w in [
            p.W_enc1, p.W_enc2, p.W_pred, p.W_trn, p.W_prec]) + \
                self.cfg.w_reg * jnp.sum(p.W_geo**2)

        return F_err + F_exp + F_trn + F_geo + F_reg

    # ------------------- Heun Integrator Step -------------------
    @partial(jax.jit, static_argnums=(0,))
    def step(self, s: State, p: Params, target: jnp.ndarray):
        s = replace(s, mu_prev=s.mu)

        loss_fn = lambda s_in, p_in: self._free_energy(s_in, p_in, target)

        # Predictor
        _, (ds1, dp) = value_and_grad(loss_fn, argnums=(0, 1))(s, p)
        s_tilde = replace(s,
            x = s.x - self.cfg.lr_s * self.cfg.dt * ds1.x,
            mu = s.mu - self.cfg.lr_s * self.cfg.dt * ds1.mu,
            r = s.r - self.cfg.lr_s * self.cfg.dt * ds1.r
        )

        # Corrector
        _, (ds2, _) = value_and_grad(loss_fn, argnums=(0, 1))(s_tilde, p)

        # Heun update
        s_new = replace(s,
            x = s.x - self.cfg.lr_s * self.cfg.dt * 0.5 * (ds1.x + ds2.x),
            mu = s.mu - self.cfg.lr_s * self.cfg.dt * 0.5 * (ds1.mu + ds2.mu),
            r = s.r - self.cfg.lr_s * self.cfg.dt * 0.5 * (ds1.r + ds2.r)
        )

        p_new = Params(
            W_geo = p.W_geo - self.cfg.lr_p * self.cfg.dt * dp.W_geo,
            W_query = p.W_query - self.cfg.lr_p * self.cfg.dt * dp.W_query,
            W_key = p.W_key - self.cfg.lr_p * self.cfg.dt * dp.W_key,
            W_enc1 = p.W_enc1 - self.cfg.lr_p * self.cfg.dt * dp.W_enc1,
            b_enc1 = p.b_enc1 - self.cfg.lr_p * self.cfg.dt * dp.b_enc1,
            W_enc2 = p.W_enc2 - self.cfg.lr_p * self.cfg.dt * dp.W_enc2,
            b_enc2 = p.b_enc2 - self.cfg.lr_p * self.cfg.dt * dp.b_enc2,
            W_pred = p.W_pred - self.cfg.lr_p * self.cfg.dt * dp.W_pred,
            b_pred = p.b_pred - self.cfg.lr_p * self.cfg.dt * dp.b_pred,
            W_trn = p.W_trn - self.cfg.lr_p * self.cfg.dt * dp.W_trn,
            b_trn = p.b_trn - self.cfg.lr_p * self.cfg.dt * dp.b_trn,
            W_prec = p.W_prec - self.cfg.lr_p * self.cfg.dt * dp.W_prec,
            b_prec = p.b_prec - self.cfg.lr_p * self.cfg.dt * dp.b_prec,
        )

        return s_new, p_new


# =====================================================================
# MAIN
# =====================================================================
def main():
    cfg = Config(n=4, d=3)
    core = GNAICore(cfg, seed=2026)

    s = core.init_state()
    p = core.init_params()

    target = jnp.ones((cfg.n, cfg.d)) * 0.5

    print("GNAI-Core (Minimal Class) コンパイル中...")
    s, p = core.step(s, p, target)
    print("✓ 単一ステップ実行成功")
    print(f"State shape: x={s.x.shape}, mu={s.mu.shape}")
    print("Model is ready for simulation loop.")


if __name__ == "__main__":
    main()