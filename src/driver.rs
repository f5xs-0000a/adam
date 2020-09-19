use rand_distr::Distribution as _;
use rand::Rng;
use rand_distr::Normal;
use crate::FLOAT;
use crate::AdamState;
use crate::AdamParams;
use itertools::Itertools as _;

////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone)]
pub struct AdamDriver {
    starting_population_size: usize,
    sustain_population_size: usize,

    vectors: Vec<Vec<FLOAT>>,
}

impl AdamDriver {
    pub fn new<R>(
        starting_population_size: usize,
        sustain_population_size: usize,
        state: &AdamState,
        rng: &mut R,
    ) -> AdamDriver
    where R: Rng {
        // generate the vectors
        let vectors = rng
            .sample_iter(rand_distr::StandardNormal)
            .chunks(state.vector_len())
            .into_iter()
            .map(|chunk| chunk.collect::<Vec<_>>())
            .take(starting_population_size)
            .collect::<Vec<_>>();

        AdamDriver {
            starting_population_size,
            sustain_population_size,
            vectors,
        }
    }

    pub fn vectors(&self) -> &[Vec<FLOAT>] {
        &*self.vectors
    }

    pub fn vectors_mut<'a>(&'a mut self) -> Vec<&'a mut [FLOAT]> {
        self.vectors.iter_mut().map(|v| &mut **v).collect::<Vec<_>>()
    }

    pub fn resample_vectors<R>(
        &mut self,
        state: &AdamState,
        params: &AdamParams,
        count: usize,
        rng: &mut R,
    ) 
    where R: Rng {
        // the mean is the vector while the variance is the v_hat.
        let mean = state.vector().to_vec();
        let mut stdev = state.v_hat(params);
        for s in stdev.iter_mut() {
            *s = (*s * params.alpha).sqrt();
        }

        println!("M:\t{:+.04} {:+.04}", mean[0], mean[1]);
        println!("S:\t{:+.04} {:+.04}", stdev[0], stdev[1]);

        self.vectors = vec![mean.clone()];
        self.vectors.reserve(count - 1);

        core::iter::repeat_with(|| {
            // sample given the mean and standard deviation
            mean.iter()
                .zip(stdev.iter())
                // take only up to how many parameters there are
                .take(mean.len())
                // create a distribution out of the mean and standard deviation
                .map(|(m, std)| Normal::new(*m, *std).unwrap())
                // sample from the distribution
                .map(|normal| normal.sample(rng))
                // collect into a vector
                .collect::<Vec<_>>()
        })
            // take up to how many is demanded minus one (since that's reserved
            // for the mean)
            .take(count - 1)
            .for_each(|vec| self.vectors.push(vec));
    }

    pub fn vector_len(&self) -> usize {
        self.vectors.first().unwrap().len()
    }

    pub fn update_vectors_and_state<R>(
        &mut self,
        mut scores: Vec<FLOAT>,
        state: &mut AdamState,
        params: &AdamParams,
        rng: &mut R,
    ) where R: Rng {
        assert!(scores.len() == self.vectors.len());

        // if we're at the start of the state
        if state.t() == 0 {
            let mut champ_index = 0;

            // set the vector to be the best of our vectors
            self.vectors
                .iter()
                .zip(scores.iter())
                .enumerate()
                .map(|(i, (v, s))| (i, v, s))
                // find the vector with the highest score
                .max_by(|(_, _, a), (_, _, b)| a.partial_cmp(b).unwrap())
                // take only the vector but also record the index of the champ
                .map(|(i, vector, _)| {
                    champ_index = i;
                    vector
                })
                .unwrap()
                // set the current vector in the state to this highest score
                // vector
                .iter()
                .zip(state.vector_mut().iter_mut())
                .for_each(|(from, to)| *to = *from);

            // swap the place of the champion to the first element so the next
            // code works. do the same for the scores.
            self.vectors.swap(0, champ_index);
            scores.swap(0, champ_index);
        }

        // generate the gradient with state.vector as the center point and using
        // its corresponding score
        // the state.vector is located in our states: it's always on the first
        // element. likewise, it should be the same for the score.
        let (vfirst, vrest) = self.vectors.split_first().unwrap();
        let (sfirst, srest) = scores.split_first().unwrap();
        let gradient = generate_gradient_at_point(
            &**vfirst,
            *sfirst,
            vrest.iter().map(|v| &**v),
            srest.iter().cloned(),
        );

        // update the state
        state.update(&*gradient, params).unwrap();

        // sample
        self.resample_vectors(
            &state,
            params,
            self.sustain_population_size,
            rng
        );
    }
}

fn generate_gradient_at_point<'a>(
    center: &[FLOAT],
    center_score: FLOAT,
    vectors: impl Iterator<Item = &'a [FLOAT]>,
    scores: impl Iterator<Item = FLOAT>,
) -> Vec<FLOAT>
{
    let mut x_diff_y_diff = vec![0.; center.len()];
    let mut x_diff_sq = vec![0.; center.len()];

    // iterate through the vectors
    for (vector, score) in vectors.zip(scores) {
        let iter = vector
            .iter()
            .zip_eq(x_diff_y_diff.iter_mut())
            .zip_eq(x_diff_sq.iter_mut())
            .zip_eq(center.iter())
            .map(|(((a, b), c), d)| (a, b, c, d));

        // for each of the parameter, increment the value of the numerator and
        // the denominator for the line of best fit
        let dy = score - center_score;
        for (param, sdxdy, sdxdx, center) in iter {
            let dx = *param - *center;
            *sdxdy += dx * dy;
            *sdxdx += dx.powi(2);
        }
    }

    // perform piecewise division
    x_diff_y_diff.into_iter()
        .zip_eq(x_diff_sq.into_iter())
        .map(|(sdxdy, sdxdx)| sdxdy / sdxdx)
        .collect::<Vec<_>>()
}
