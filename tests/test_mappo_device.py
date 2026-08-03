import numpy as np
import pytest
import torch

from training.mappo_buffer import AsyncMAPPOBuffer, AsyncTransition
from training.mappo_networks import CandidateScoringActor, CentralizedCritic


def _transition():
    return AsyncTransition(agent_id="assignment", local_obs=[1.0, 2.0], global_state=[3.0, 4.0],
        action=0, action_mask=[True], candidate_features=[[5.0]], candidate_feature_names=("x",),
        log_prob=0.0, value=0.0, reward=1.0, done=True, next_global_state=[3.0, 4.0],
        event_type="PARCEL_RELEASE", event_time=0.0)


def test_cpu_actor_critic_inference_and_buffer_boundary():
    device=torch.device("cpu")
    actor=CandidateScoringActor(2, 1, [4]).to(device)
    critic=CentralizedCritic(2, [4]).to(device)
    action, log_prob=actor.act([1, 2], 0, [[3]], [True])
    with torch.inference_mode(): value=float(critic(torch.tensor([1, 2], dtype=torch.float32, device=device)).item())
    buffer=AsyncMAPPOBuffer(); item=_transition(); item.action=action; item.log_prob=log_prob; item.value=value; buffer.append(item)
    assert next(actor.parameters()).device == next(critic.parameters()).device == device
    assert isinstance(buffer.transitions[0].log_prob, float)
    assert not any(isinstance(value, torch.Tensor) for value in vars(buffer.transitions[0]).values())
    buffer.compute_returns_and_advantages(); assert isinstance(buffer.advantages, np.ndarray)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_cuda_actor_critic_update_and_cpu_portable_checkpoint(tmp_path):
    device=torch.device("cuda")
    actor=CandidateScoringActor(2, 1, [4]).to(device); critic=CentralizedCritic(2, [4]).to(device)
    obs=torch.tensor([[1.,2.]],device=device); events=torch.tensor([0],device=device)
    candidates=torch.tensor([[[3.]]],device=device); masks=torch.tensor([[True]],device=device)
    actions=torch.tensor([0],device=device)
    log_prob,_=actor.evaluate_actions(obs,events,candidates,masks,actions)
    loss=-log_prob.mean()+critic(obs).square().mean(); loss.backward()
    path=tmp_path/"smoke.pt"; torch.save({"actor":actor.state_dict(),"critic":critic.state_dict()},path)
    loaded=torch.load(path,map_location="cpu",weights_only=False)
    assert all(t.device.type == "cpu" for state in loaded.values() for t in state.values())
    print("CUDA_NEURAL_TRAINING_SMOKE_VALID")
