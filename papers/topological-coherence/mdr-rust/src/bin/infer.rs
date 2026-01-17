//! MDR Inference Binary

use clap::Parser;
use mdr::{
    model::MDRModel,
    tokenizer::SimpleTokenizer,
};
use std::io::{self, BufRead, Write};
use tracing::info;

#[derive(Parser, Debug)]
#[command(name = "mdr-infer")]
#[command(about = "Run inference with an MDR model")]
struct Args {
    /// Path to model file
    #[arg(short, long)]
    model: String,

    /// Input prompt (if not provided, runs interactive mode)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Maximum tokens to generate
    #[arg(long, default_value_t = 100)]
    max_tokens: usize,

    /// Temperature for sampling
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// Interactive mode
    #[arg(short, long)]
    interactive: bool,
}

fn main() {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Loading model from: {}", args.model);
    let mut model = MDRModel::load(&args.model).expect("Failed to load model");

    let stats = model.stats();
    println!("Loaded MDR Model");
    println!("================");
    println!("Parameters: {}", stats.total_params);
    println!("Layers: {} ({} with topology)", stats.total_layers, stats.topology_layers);
    println!();

    let tokenizer = SimpleTokenizer::ascii();

    if args.interactive || args.prompt.is_none() {
        // Interactive mode
        println!("Interactive mode. Type 'quit' to exit.");
        println!();

        let stdin = io::stdin();
        let mut stdout = io::stdout();

        loop {
            print!("> ");
            stdout.flush().unwrap();

            let mut line = String::new();
            stdin.lock().read_line(&mut line).unwrap();
            let line = line.trim();

            if line == "quit" || line == "exit" {
                break;
            }

            if line.is_empty() {
                continue;
            }

            // Generate
            let prompt_tokens = tokenizer.encode(line);
            let prompt_tokens = &prompt_tokens[..prompt_tokens.len() - 1];  // Remove EOS

            model.reset_context();
            let output = model.generate(prompt_tokens, args.max_tokens, args.temperature);
            let text = tokenizer.decode(&output);

            println!("{}", text);
            println!();
        }
    } else if let Some(prompt) = args.prompt {
        // Single prompt mode
        let prompt_tokens = tokenizer.encode(&prompt);
        let prompt_tokens = &prompt_tokens[..prompt_tokens.len() - 1];  // Remove EOS

        model.reset_context();
        let output = model.generate(prompt_tokens, args.max_tokens, args.temperature);
        let text = tokenizer.decode(&output);

        println!("{}", text);
    }
}
