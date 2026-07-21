import numpy as np
from training.mappo_buffer import AsyncMAPPOBuffer, AsyncTransition

def tr(agent, reward, t):
    return AsyncTransition(agent,[1],[1],0,[True],[[0]],("x",),0,0,reward,False,[1],"PARCEL_RELEASE",t)

def test_per_agent_advantages_normalized_separately():
    b=AsyncMAPPOBuffer()
    for x in [1,2,100,200]: b.append(tr("assignment" if x<100 else "bus", x, x))
    b.compute_returns_and_advantages(gamma=0.0, gae_lambda=0.95, per_agent_normalize=True)
    a=b.advantages[:2]; bus=b.advantages[2:]
    assert np.isclose(a.mean(),0) and np.isclose(bus.mean(),0)
    assert np.isclose(a.std(),1) and np.isclose(bus.std(),1)
