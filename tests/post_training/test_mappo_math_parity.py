import numpy as np
import torch
from training.mappo_buffer import AsyncMAPPOBuffer, AsyncTransition
from training.post_training.trainer import clipped_policy_loss


def _transition(reward, value, event_time, done=False):
    return AsyncTransition(agent_id="assignment", local_obs=[0.], global_state=[0.], action=0,
        action_mask=[True], candidate_features=[[0.]], candidate_feature_names=("x",),
        log_prob=0., value=value, reward=reward, done=done, next_global_state=[0.],
        event_type="PARCEL_RELEASE", event_time=event_time)


def test_event_discount_td_gae_and_return_match_legacy_equations():
    buf=AsyncMAPPOBuffer(); buf.append(_transition(1., .4, 0)); buf.append(_transition(2., .5, 10, True))
    returns,_=buf.compute_returns_and_advantages(.9,.8,reference_time_unit=5,per_agent_normalize=False)
    discount=.9**2
    delta1=2-.5
    delta0=1+discount*.5-.4
    expected_adv0=delta0+discount*.8*delta1
    np.testing.assert_allclose(returns, [expected_adv0+.4, delta1+.5], rtol=1e-6)


def test_ratio_clip_entropy_value_and_gradient_clip_parity():
    new=torch.tensor([.3,-.4]); old=torch.tensor([.1,-.2]); adv=torch.tensor([1.,-1.])
    ratio=(new-old).exp(); surrogate=-torch.minimum(ratio*adv, ratio.clamp(.8,1.2)*adv).mean()
    assert torch.allclose(clipped_policy_loss(new,old,adv,.2),surrogate)
    entropy=torch.tensor([.5,.7]); coef=.01
    assert torch.allclose(surrogate-coef*entropy.mean(), clipped_policy_loss(new,old,adv,.2)-coef*entropy.mean())
    values=torch.tensor([1.,2.]); returns=torch.tensor([1.5,1.])
    assert torch.allclose(torch.nn.functional.mse_loss(values,returns), ((values-returns)**2).mean())
    parameter=torch.nn.Parameter(torch.tensor([3.,4.])); parameter.grad=torch.tensor([3.,4.])
    torch.nn.utils.clip_grad_norm_([parameter], 1.)
    assert parameter.grad.norm() <= 1.000001
