//! MDR Training Binary

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use mdr::{
    model::{MDRConfig, MDRModel},
    training::{Trainer, TrainingConfig, TrainingExample, DataLoader, Batch},
    tokenizer::SimpleTokenizer,
};
use std::fs;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "mdr-train")]
#[command(about = "Train an MDR (Meta Distributed Representations) model")]
struct Args {
    /// Model size: small, medium, or large
    #[arg(short, long, default_value = "small")]
    size: String,

    /// Training data file (text)
    #[arg(short, long)]
    data: Option<String>,

    /// Output directory for checkpoints
    #[arg(short, long, default_value = "checkpoints")]
    output: String,

    /// Number of training steps
    #[arg(long, default_value_t = 1000)]
    steps: usize,

    /// Batch size
    #[arg(long, default_value_t = 4)]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value_t = 1e-4)]
    lr: f32,

    /// Evaluation interval
    #[arg(long, default_value_t = 100)]
    eval_interval: usize,

    /// Checkpoint interval
    #[arg(long, default_value_t = 500)]
    checkpoint_interval: usize,

    /// Resume from checkpoint
    #[arg(long)]
    resume: Option<String>,
}

fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("MDR Training");
    info!("============");

    // Create model config
    let model_config = match args.size.as_str() {
        "small" => MDRConfig::small(),
        "medium" => MDRConfig::medium(),
        "large" => MDRConfig::large(),
        _ => {
            warn!("Unknown size '{}', using small", args.size);
            MDRConfig::small()
        }
    };

    info!("Model config: {:?}", model_config);
    info!("Parameters: {}", model_config.param_count());
    info!("Size: {:.2} MB", model_config.size_bytes() as f32 / 1e6);

    // Create or load model
    let model = if let Some(ref checkpoint_path) = args.resume {
        info!("Resuming from checkpoint: {}", checkpoint_path);
        match Trainer::load_checkpoint(checkpoint_path) {
            Ok(trainer) => {
                info!("Resumed at step {}", trainer.step);
                trainer.model
            }
            Err(e) => {
                warn!("Failed to load checkpoint: {}", e);
                info!("Creating new model");
                MDRModel::new(model_config)
            }
        }
    } else {
        info!("Creating new model");
        MDRModel::new(model_config)
    };

    // Print model stats
    let stats = model.stats();
    println!("{}", stats);

    // Training config
    let train_config = TrainingConfig {
        learning_rate: args.lr,
        batch_size: args.batch_size,
        num_steps: args.steps,
        eval_interval: args.eval_interval,
        checkpoint_interval: args.checkpoint_interval,
        ..Default::default()
    };

    // Create trainer
    let mut trainer = Trainer::new(model, train_config);

    // Load or generate training data
    let examples = if let Some(ref data_path) = args.data {
        load_training_data(data_path)
    } else {
        info!("No data file provided, using synthetic data");
        generate_synthetic_data(100)
    };

    info!("Training examples: {}", examples.len());

    let mut data_loader = DataLoader::new(examples, args.batch_size);

    // Create output directory
    fs::create_dir_all(&args.output).expect("Failed to create output directory");

    // Training loop
    let pb = ProgressBar::new(args.steps as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) loss: {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    while trainer.step < args.steps {
        // Get batch
        let batch = match data_loader.next_batch() {
            Some(b) => b,
            None => {
                data_loader.shuffle();
                continue;
            }
        };

        // Training step
        let loss = trainer.train_step(&batch);

        // Update progress
        pb.set_position(trainer.step as u64);
        pb.set_message(format!("{:.4}", trainer.avg_loss()));

        // Evaluation
        if trainer.step % args.eval_interval == 0 {
            info!("Step {}: loss = {:.4}", trainer.step, trainer.avg_loss());

            // Generate sample
            let sample = generate_sample(&mut trainer.model);
            info!("Sample: {}", sample);
        }

        // Checkpoint
        if trainer.step % args.checkpoint_interval == 0 {
            let checkpoint_path = format!("{}/checkpoint_{}.bin", args.output, trainer.step);
            match trainer.save_checkpoint(&checkpoint_path) {
                Ok(_) => info!("Saved checkpoint: {}", checkpoint_path),
                Err(e) => warn!("Failed to save checkpoint: {}", e),
            }
        }
    }

    pb.finish_with_message("Training complete!");

    // Save final model
    let final_path = format!("{}/model_final.bin", args.output);
    match trainer.model.save(&final_path) {
        Ok(_) => info!("Saved final model: {}", final_path),
        Err(e) => warn!("Failed to save final model: {}", e),
    }
}

fn load_training_data(path: &str) -> Vec<TrainingExample> {
    let tokenizer = SimpleTokenizer::ascii();

    let text = fs::read_to_string(path).expect("Failed to read training data");

    // Split into chunks
    let chunk_size = 128;
    let tokens = tokenizer.encode(&text);

    let mut examples = Vec::new();
    for chunk in tokens.chunks(chunk_size) {
        if chunk.len() < 2 {
            continue;
        }
        examples.push(TrainingExample {
            input_tokens: chunk[..chunk.len() - 1].to_vec(),
            target_tokens: chunk[1..].to_vec(),
        });
    }

    examples
}

fn generate_synthetic_data(n: usize) -> Vec<TrainingExample> {
    // Simple patterns for testing
    let patterns = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Hello, world! This is a test.",
        "One two three four five six seven eight nine ten.",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "The answer to life, the universe, and everything is 42.",
    ];

    let tokenizer = SimpleTokenizer::ascii();

    let mut examples = Vec::new();
    for i in 0..n {
        let text = patterns[i % patterns.len()];
        let tokens = tokenizer.encode(text);

        if tokens.len() < 2 {
            continue;
        }

        examples.push(TrainingExample {
            input_tokens: tokens[..tokens.len() - 1].to_vec(),
            target_tokens: tokens[1..].to_vec(),
        });
    }

    examples
}

fn generate_sample(model: &mut MDRModel) -> String {
    let tokenizer = SimpleTokenizer::ascii();

    let prompt = tokenizer.encode("The ");
    let prompt = &prompt[..prompt.len() - 1];  // Remove EOS

    model.reset_context();
    let output = model.generate(prompt, 20, 0.8);

    tokenizer.decode(&output)
}
