import numpy as np, torch
from training.mappo_buffer import AsyncMAPPOBuffer, AsyncTransition
from training.mappo_trainer import _padded_candidate_batch

def test_minibatch_keeps_event_ids_aligned():
    b=AsyncMAPPOBuffer()
    for eid,event in [(0,"PARCEL_RELEASE"),(1,"TRUCK_AVAILABLE")]:
        agent="assignment" if eid==0 else "truck"
        b.append(AsyncTransition(agent,[float(eid)],[0.],0,[True],[[0.]],("x",),0.,0.,0.,False,[0.],event,float(eid),eid,total_reward=0.))
    idx=next(b.minibatch_indices(2, np.random.default_rng(1)))
    ids=torch.tensor([b.transitions[i].event_type_id for i in idx])
    obs=torch.tensor([b.transitions[i].local_obs[0] for i in idx])
    assert set(ids.tolist())==set(obs.int().tolist())
