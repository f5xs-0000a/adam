mod params;
mod state;

pub use params::AdamParams;
pub use state::AdamState;

#[cfg(feature = "FLOAT32")]
type FLOAT = f32;

#[cfg(not(feature = "FLOAT32"))]
type FLOAT = f64;

#[cfg(feature = "gitversion")]
pub fn git_version() -> &'static str {
    git_version::git_version!()
}
