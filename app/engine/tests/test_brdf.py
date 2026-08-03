import numpy as np
import taichi as ti

from tracer.sampling.brdf import BRDF

# ti.init() living here (rather than in brdf.py itself) is the one
# exception to "don't call ti.init() outside the process entry point" --
# a test module IS its own entry point when run standalone. CPU backend is
# enough for these and doesn't require a CUDA device to be present in CI.
ti.init(arch=ti.cpu)

brdf = BRDF()


@ti.kernel
def _evaluate(albedo: ti.math.vec3) -> ti.math.vec3:  # type: ignore
    return brdf.evaluate(albedo)


@ti.kernel
def _pdf(cos_theta: ti.f32) -> ti.f32:  # type: ignore
    return brdf.pdf(cos_theta)  


@ti.kernel
def _sample_bounce(albedo: ti.math.vec3, cos_theta: ti.f32, l_i: ti.math.vec3) -> ti.math.vec3: # type: ignore
    return brdf.sample_cosine_weighted_bounce(albedo, cos_theta, l_i)


def test_reciprocity():
    # Lambertian f_r doesn't depend on wi/wo at all -- trivial reciprocity
    # here, but this is the same property GGX (Phase 5) has to satisfy once
    # f_r actually depends on direction.
    albedo = np.array([0.6, 0.3, 0.1], dtype=np.float32)
    result = np.array(_evaluate(albedo))
    assert np.allclose(result, albedo / np.pi, atol=1e-6)


def test_zero_variance_under_cosine_sampling():
    # With l_i held constant and cos_theta drawn from exactly the pdf this
    # BRDF's own importance sampling produces, every single sample of
    # sample_cosine_weighted_bounce should independently equal albedo * l_i
    # -- not just the average over many samples. Cosine-weighted sampling
    # paired with a Lambertian BRDF is the "perfect importance sampling"
    # case: the estimator has zero variance. If even one sample is off,
    # the cancellation is wrong, not just noisy -- a much stronger check
    # than averaging a batch of random samples.
    albedo = np.array([0.8, 0.5, 0.2], dtype=np.float32)
    l_i = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    for cos_theta in np.linspace(0.01, 1.0, 20):
        result = np.array(_sample_bounce(albedo, float(cos_theta), l_i))
        assert np.allclose(result, albedo * l_i, atol=1e-5)


def test_backface_returns_zero():
    albedo = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    l_i = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    assert _pdf(-0.1) == 0.0
    assert np.allclose(np.array(_sample_bounce(albedo, -0.1, l_i)), [0.0, 0.0, 0.0])


def test_grazing_angle_no_nan():
    albedo = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    l_i = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    result = np.array(_sample_bounce(albedo, 0.0, l_i))
    assert not np.any(np.isnan(result))


def test_pdf_at_normal_incidence():
    assert abs(_pdf(1.0) - (1.0 / np.pi)) < 1e-6