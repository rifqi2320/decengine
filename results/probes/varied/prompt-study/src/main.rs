use anyhow::Result;
use clap::Parser;
use decengine_models::{ModelRegistry, ModelStore};
use std::path::PathBuf;

mod exporter_base;

#[derive(Parser)]
struct Args {
    #[arg(long)] home: PathBuf,
    #[arg(long)] input: PathBuf,
    #[arg(long)] output: PathBuf,
    #[arg(long)] model: String,
    #[arg(long)] variant: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let registry = ModelRegistry::bundled()?;
    let store = ModelStore::new(args.home, registry);
    exporter_base::export(&store, &args.model, &args.input, &args.output, &args.variant)
}
