import pyglobalsearch as gs

print(gs.__doc__)
print(dir(gs))

nelder_mead_config = gs.builders.nelder_mead(
    simplex_delta=1e-3,
    sd_tolerance=1e-3,
    max_iter=100,
    alpha=1.0,
    gamma=2.0,
    rho=0.5,
    sigma=0.5,
)

print(nelder_mead_config)
print(nelder_mead_config.simplex_delta)

# create a hagerzhang config
hagerzhang_config = gs.builders.hagerzhang(
    delta=1e-3,
    sigma=1e-3,
    epsilon=1e-3,
    theta=1e-3,
    gamma=1e-3,
    eta=1e-3,
    bounds=[0.0, 1.0],
)

print(hagerzhang_config)
print(hagerzhang_config.delta)

lbfgs_config = gs.builders.lbfgs(
    max_iter=100,
    tolerance_grad=1e-3,
    tolerance_cost=1e-6,
    history_size=5,
    line_search_params=hagerzhang_config,
)
