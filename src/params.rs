use crate::FLOAT;

////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, PartialEq)]
pub struct AdamParams {
    pub alpha:   FLOAT,
    pub epsilon: FLOAT,
    pub beta_1:  FLOAT,
    pub beta_2:  FLOAT,
}

impl Default for AdamParams {
    fn default() -> Self {
        AdamParams {
            alpha:   0.001,
            epsilon: 0.00000001,
            beta_1:  0.9,
            beta_2:  0.999,
        }
    }
}
