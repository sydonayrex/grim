//! Cli handler for `grim spec ...` commands.

use grim_core::error::Result;

pub fn cmd_spec_train(target: String, output: String, dataset: String) -> Result<()> {
    grim_speculative::train_speculative_draft(&target, &output, &dataset)?;
    Ok(())
}
