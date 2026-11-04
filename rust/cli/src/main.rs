mod args;
mod bench;
mod output;
mod run;
mod stats_table;
#[cfg(test)]
mod tests;
mod viability;

use std::env;
use std::process::ExitCode;

use args::{Command, parse_args, print_usage};

fn main() -> ExitCode {
    match parse_args(env::args().skip(1)) {
        Ok(Some(Command::Run(options))) => run::run_command(options),
        Ok(Some(Command::Bench(options))) => bench::bench_command(options),
        Ok(Some(Command::Inspect(options))) => run::inspect_command(options),
        Ok(Some(Command::Viability(options))) => viability::viability_command(options),
        Ok(None) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            print_usage();
            ExitCode::from(2)
        }
    }
}
