use crate::{
    params::AdamParams,
    FLOAT,
};
use thiserror::Error;

////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamState {
    m: Vec<FLOAT>, // has a length of d
    v: Vec<FLOAT>, // has a length of d
    pub t: i32,
    vector: Vec<FLOAT>, // has a length of d
}

#[derive(Clone, Copy, Debug, Error)]
#[error("Parameter count is unequal")]
pub struct ParamCountError;

impl AdamState {
    pub fn new(param_count: usize) -> AdamState {
        AdamState {
            m:       vec![0.; param_count],
            v:       vec![0.; param_count],
            t:       0,
            vector: vec![0.; param_count],
        }
    }

    pub fn vector_len(&self) -> usize {
        self.m.len()
    }

    pub fn reset_state(&mut self) {
        let count = self.vector_len();
        self.m = vec![0.; count];
        self.v = vec![0.; count];
        self.t = 0;
    }

    pub fn vector(&self) -> &[FLOAT] {
        &*self.vector
    }

    pub fn vector_mut(&mut self) -> &mut [FLOAT] {
        &mut self.vector
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

    pub fn m_hat(&self, params: &AdamParams) -> Vec<FLOAT> {
        get_bias_corrected_moment_estimate(&self.m, params.beta_1, self.t)
    }

    pub fn v_hat(&self, params: &AdamParams) -> Vec<FLOAT> {
        get_bias_corrected_moment_estimate(&self.v, params.beta_2, self.t)
    }

    pub fn update(
        &mut self,
        gradient: &[FLOAT],
        params: &AdamParams,
    ) -> Result<(), ParamCountError>
    {
        if gradient.len() != self.m.len() {
            Err(ParamCountError)?;
        }

        self.t += 1;

        update_biased_moment_estimate(
            &mut self.m,
            gradient,
            params.beta_1,
            true,
        );

        update_biased_moment_estimate(
            &mut self.v,
            gradient,
            params.beta_2,
            false,
        );

        let m_hat = self.m_hat(params);
        let v_hat = self.v_hat(params);

        // update the current vector
        let iter = self.vector.iter_mut().zip(m_hat.iter()).zip(v_hat.iter());
        for ((param, mean), var) in iter {
            *param -= params.alpha * *mean / (var.sqrt() + params.epsilon);
        }

        Ok(())
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
    is_mean: bool
)
{
    let power = match is_mean {
        true => 1,
        false => 2,
    };

    for (me_val, g_val) in me.iter_mut().zip(gradient.iter()) {
        *me_val = beta * *me_val + (1. - beta) * g_val.powi(power)
    }
}
