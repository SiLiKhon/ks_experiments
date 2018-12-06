import numpy as np
import matplotlib.pyplot as plt
from weighted_ks import ks_2samp_w as ks_2samp
from tqdm import tqdm

bins_ks = np.linspace(0, 0.6, 301)

def experiment(**kwargs):
    N1 = kwargs['N1']
    N2 = kwargs['N2']
    num_iterations = kwargs['num_iterations']
    mode = kwargs['mode']
    generator_1 = kwargs['generator_1']
    generator_2 = kwargs['generator_2']
    w1_func = kwargs['w1_func']
    w2_func = kwargs['w2_func']

    ks_stat = []
    if mode == 'bootstrap':
        sample1_base = generator_1([N1], **kwargs)
        sample2_base = generator_2([N2], **kwargs)
        w1_base = w1_func(sample1_base)
        w2_base = w2_func(sample2_base)

    #for _ in tqdm(range(num_iterations)):
    for i in range(num_iterations):
        if mode == 'true':
            sample1 = generator_1([N1], **kwargs)
            sample2 = generator_2([N2], **kwargs)
            w1 = w1_func(sample1)
            w2 = w2_func(sample2)
        elif mode == 'bootstrap':
            i1 = np.random.choice(np.arange(len(sample1_base)), size=len(sample1_base))
            i2 = np.random.choice(np.arange(len(sample2_base)), size=len(sample2_base))
            sample1 = sample1_base[i1]
            sample2 = sample2_base[i2]
            w1 = w1_base[i1]
            w2 = w2_base[i2]
        else:
            raise NotImplementedError(mode)
        if i==0:
            print("{}, {}".format(w1.sum(), w2.sum()))
        stat = ks_2samp(sample1, sample2, w1, w2)
        ks_stat.append(stat)

    return np.array(ks_stat)

def params(**kwargs):
    result = dict(
                N1 = 1000,
                N2 = 1000,
                num_iterations = 10000,
                mode = "true",
                mean2 = 0.0,
                generator_1 = lambda size, **kwargs: np.random.normal(size=size),
                generator_2 = lambda size, **kwargs: np.random.normal(size=size, loc=kwargs['mean2']),
                w1_func = lambda x: np.ones_like(x),
                w2_func = lambda x: np.ones_like(x)
            )
    for k, v in kwargs.items():
        result[k] = v
    return result

def abnormal(size, loc=0.0, scale=1.0, drop_l=0.0, drop_r=0.5, drop_f=0.1, **kwargs):
    data = np.random.normal(loc=loc, scale=scale, size=size)
    drop_candidates = (data >= drop_l) & (data < drop_r)
    keep_candidates = ~drop_candidates
    drop_candidates &= (np.random.uniform(size=size) > drop_f)
    data = data[keep_candidates | drop_candidates]
    return data




fig, (ax0, ax1) = plt.subplots(2, 1)


ax0.hist(experiment(**params(mean2=0.1)), bins=bins_ks, label='without weights', alpha=0.4)
ax1.hist(experiment(**params(mean2=0.1,
                      mode='bootstrap')), bins=bins_ks, label='without weights', alpha=0.4)
for s in [0.1, 0.5, 1.0, 5.0]:
    ax0.hist(experiment(**params(
                            mean2=0.1,
                            w1_func=lambda x: np.random.normal(loc=1., scale=s, size=x.shape),
                            w2_func=lambda x: np.random.normal(loc=1., scale=s, size=x.shape)
                        )), bins=bins_ks, label='normal weights, s={}'.format(s), histtype='step')
    ax1.hist(experiment(**params(
                            mode='bootstrap',
                            mean2=0.1,
                            w1_func=lambda x: np.random.normal(loc=1., scale=s, size=x.shape),
                            w2_func=lambda x: np.random.normal(loc=1., scale=s, size=x.shape)
                        )), bins=bins_ks, label='normal weights, s={}'.format(s), histtype='step')

ax0.legend()
ax0.set_title('true ks')
ax1.legend()
ax1.set_title('bootstrap ks')
fig.tight_layout()

ax0.text(0.4, 0.98,
        """Comparing N(0.0, 1.0)
with N(0.1, 1.0)

For weighted samples
weights are distributed
as N(1, s)

Sample sizes: {} and {}""".format(params()['N1'], params()['N2']),
         horizontalalignment='left',
         verticalalignment='top', transform=ax0.transAxes,
         fontsize=6)
fig.savefig('result.pdf')
plt.show()