use crate::{
    params::AdamParams,
    FLOAT,
};
use thiserror::Error;

////////////////////////////////////////////////////////////////////////////////

pub struct AdamState {
    m: Vec<FLOAT>, // has a length of d
    v: Vec<FLOAT>, // has a length of d
    t: i32,

    vectors: Vec<Vec<FLOAT>>, // has a length of n * d
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("Parameter count is unequal")]
    ParamCountError,

    #[error("The vector list is empty")]
    EmptyVectorList,
}

impl AdamState {
    pub fn new(initial_vectors: Vec<Vec<FLOAT>>) -> Result<AdamState, Error> {
        assert_equal_parameter_count(initial_vectors.iter())?;

        let mut state = AdamState {
            m:       vec![],
            v:       vec![],
            t:       0,
            vectors: initial_vectors,
        };

        let count = state.get_parameter_count();
        state.m = vec![0.; count];
        state.v = vec![0.; count];

        Ok(state)
    }

    pub fn get_vectors<'a>(&'a self) -> &'a [Vec<FLOAT>] {
        &self.vectors
    }

    pub fn update_vectors(
        &mut self,
        vectors: Vec<Vec<FLOAT>>,
    ) -> Result<(), Error>
    {
        assert_equal_parameter_count_given_count(
            self.get_parameter_count(),
            vectors.iter(),
        )?;

        self.vectors = vectors;
        Ok(())
    }

    pub fn get_parameter_count(&self) -> usize {
        self.vectors.first().map(|x| x.len()).unwrap()
    }

    pub fn reset_state(&mut self) {
        let count = self.get_parameter_count();
        self.m = vec![0.; count];
        self.v = vec![0.; count];
        self.t = 0;
    }

    pub fn m(&self) -> &[FLOAT] {
        &self.m
    }

    pub fn v(&self) -> &[FLOAT] {
        &self.v
    }

    pub fn t(&self) -> i32 {
        self.t
    }

    pub fn update(
        &mut self,
        scores: Vec<FLOAT>,
        params: &AdamParams,
    )
    {
        self.t += 1;

        // generate the gradients
        let gradients = generate_gradients(
            self.vectors.iter().map(|vec| &**vec),
            scores.iter().cloned(),
        );

        update_biased_moment_estimate(&mut self.m, &gradients, params.beta_1);

        update_biased_moment_estimate(&mut self.v, &gradients, params.beta_2);

        let m_hat =
            get_bias_corrected_moment_estimate(&self.m, params.beta_1, self.t);

        let v_hat =
            get_bias_corrected_moment_estimate(&self.v, params.beta_2, self.t);

        generate_new_vectors_in_place(
            &mut self.vectors,
            m_hat,
            v_hat,
            params.alpha,
            params.epsilon,
        )
    }
}

fn assert_equal_parameter_count<'a>(
    mut vectors: impl Iterator<Item = &'a Vec<FLOAT>>
) -> Result<(), Error> {
    let count = match vectors.next() {
        Some(vector) => vector.len(),
        None => Err(Error::EmptyVectorList)?,
    };

    assert_equal_parameter_count_given_count(count, vectors)
}

fn assert_equal_parameter_count_given_count<'a>(
    count: usize,
    vectors: impl Iterator<Item = &'a Vec<FLOAT>>,
) -> Result<(), Error>
{
    for vector in vectors {
        if vector.len() != count {
            return Err(Error::ParamCountError);
        }
    }

    Ok(())
}

fn generate_new_vectors_in_place(
    vectors: &mut [Vec<FLOAT>],
    m_hat: Vec<FLOAT>,
    v_hat: Vec<FLOAT>,
    alpha: FLOAT,
    epsilon: FLOAT,
)
{
    for vector in vectors.iter_mut() {
        let iter = vector.iter_mut().zip(m_hat.iter()).zip(v_hat.iter());

        for ((param, m_val), v_val) in iter {
            *param -= alpha * m_val / (v_val.sqrt() + epsilon);
        }
    }
}

fn get_bias_corrected_moment_estimate(
    me: &[FLOAT],
    beta: FLOAT,
    t: i32,
) -> Vec<FLOAT>
{
    me.iter()
        .map(|me_val| me_val / (1. - beta.powi(t as i32)))
        .collect::<Vec<_>>()
}

fn update_biased_moment_estimate(
    me: &mut [FLOAT],
    gradient: &[FLOAT],
    beta: FLOAT,
)
{
    for (me_val, g_val) in me.iter_mut().zip(gradient.iter()) {
        *me_val = beta * *me_val + (1. - beta) * *g_val
    }
}

fn generate_gradients<'a>(
    param_iter: impl Iterator<Item = &'a [FLOAT]>,
    scores_iter: impl Iterator<Item = FLOAT>,
) -> Vec<FLOAT>
{
    let mut x_counter = 0;
    let mut param_iter = param_iter.inspect(|_| x_counter += 1);

    // create a vector of sums of the parameters
    let mut x_sums = param_iter.next().map(|vec| vec.to_vec()).unwrap();

    // sum all each of the parameters for all the samples
    for vec in param_iter {
        for (param, sum) in vec.iter().zip(x_sums.iter_mut()) {
            *sum += *param;
        }
    }

    // sum all the y's
    let mut y_counter = 0;
    let rise = scores_iter.inspect(|_| y_counter += 1).sum::<FLOAT>();

    // test if they're the same length
    assert!(
        x_counter != y_counter,
        "The number of samples do not equal the number of scores!"
    );

    for runs in x_sums.iter_mut() {
        *runs = rise / *runs;
    }
    let gradients = x_sums;

    gradients
}
