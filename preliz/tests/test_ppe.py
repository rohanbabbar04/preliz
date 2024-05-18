import numpy as np
import pandas as pd
import preliz as pz
import pymc as pm
import pytest
from numpy.testing import assert_allclose

np.random.seed(42)


@pytest.mark.parametrize(
    "params",
    [
        {
            "mu_x": 0,
            "sigma_x": 10,
            "sigma_z": 10,
            "target": pz.Normal(mu=174, sigma=20),
            "X": np.random.normal(0, 10, 120),
            "new_mu_x": 174.015898,
            "new_sigma_x": 1.971567,
            "new_sigma_z": 19.856088,
        },
        {
            "mu_x": [0, 1],
            "sigma_x": [10, 10],
            "sigma_z": 10,
            "target": pz.Normal(mu=174, sigma=20),
            "X": np.random.normal(0, 10, 120),
            "new_mu_x": (174.211816, 173.822841),
            "new_sigma_x": (2.621553, 2.691291),
            "new_sigma_z": 19.769245,
        },
        {
            "mu_x": 0,
            "sigma_x": 10,
            "sigma_z": 10,
            "target": [(pz.Normal(mu=174, sigma=20), 0.5), (pz.Normal(mu=176, sigma=19.5), 0.5)],
            "X": np.random.normal(0, 10, 120),
            "new_mu_x": 175.015421,
            "new_sigma_x": 1.941163,
            "new_sigma_z": 19.63016,
        },
        {
            "mu_x": [0, 1],
            "sigma_x": [10, 10],
            "sigma_z": 10,
            "target": [
                (pz.Normal(mu=174, sigma=20), 0.5),
                (pz.Normal(mu=176, sigma=19.5), 0.4),
                (pz.StudentT(mu=174, sigma=20, nu=3), 0.1),
            ],
            "X": np.random.normal(0, 10, 120),
            "new_mu_x": [175.045326, 174.626407],
            "new_sigma_x": [2.933791, 2.917447],
            "new_sigma_z": 21.413898,
        },
    ],
)
def test_ppe(params):
    Y = 2 + np.random.normal(params["X"], 1)
    target = params.pop("target")
    with pm.Model() as model:
        x = pm.Normal("x", mu=params["mu_x"], sigma=params["sigma_x"])
        z = pm.HalfNormal("z", params["sigma_z"])
        x_idx = (
            x
            if max(np.asarray(params["mu_x"]).size, np.asarray(params["sigma_x"]).size) == 1
            else x[np.repeat(np.arange(2), Y.size / 2)]
        )
        y = pm.Normal("y", x_idx, z, observed=Y)
    prior, new_prior, pymc_string = pz.ppe(model, target)
    assert_allclose(new_prior["x"].mu, params["new_mu_x"], 1e-6)
    assert_allclose(new_prior["x"].sigma, params["new_sigma_x"], 1e-6)
    assert_allclose(new_prior["z"].sigma, params["new_sigma_z"], 1e-6)


@pytest.mark.parametrize(
    "params",
    [
        {
            "mu_a": 0,
            "sigma_a": 10,
            "sigma_b": 10,
            "sigma_z": 10,
            "target": pz.Normal(mu=40, sigma=7),
            "new_mu_a": 40.013092,
            "new_sigma_a": 0.16864,
            "new_sigma_b": 6.991618,
            "new_sigma_z": 6.991618,
        },
    ],
)
def test_ppe_testdata(params):
    csv_data = pd.read_csv(r"testdata/chemical_shifts_theo_exp.csv")
    diff = csv_data.theo - csv_data.exp
    cat_encode = pd.Categorical(csv_data["aa"])
    idx = cat_encode.codes
    coords = {"aa": cat_encode.categories}
    with pm.Model(coords=coords) as model:
        # hyper_priors
        a = pm.Normal("a", mu=0, sigma=10)
        b = pm.HalfNormal("b", 10)
        z = pm.HalfNormal("z", sigma=10)

        x = pm.Normal("x", mu=a, sigma=b, dims="aa")  # No prior for this!

        y = pm.Normal("y", mu=x[idx], sigma=z, observed=diff)
    prior, new_prior, pymc_string = pz.ppe(model, params["target"])
    assert_allclose(new_prior["a"].mu, params["new_mu_a"], 1e-6)
    assert_allclose(new_prior["a"].sigma, params["new_sigma_a"], 1e-6)
    assert_allclose(new_prior["b"].sigma, params["new_sigma_b"], 1e-6)
    assert_allclose(new_prior["z"].sigma, params["new_sigma_z"], 1e-6)
