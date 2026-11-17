//! Writes a wing's facts header from its card (`gen_facts_header <card.json> <header>`) or the rows header of a door build from the
//! cards of its targets (`gen_facts_header --door-rows <host card> [<device card>] <header>`). With `--check` first, compares the
//! header on disk with what the cards make and exits 1 on a difference.
use raptor_wings::card::{door_rows_header, facts_header, read_card, Card};
use std::process::ExitCode;

/// Reads the card in the file.
fn card_of(path: &str) -> Result<Card, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    read_card(&text)
}

/// The header the arguments ask for: a wing's facts header from its card, or with `--door-rows` the rows header of a door build from
/// the host card and, when given, the device card. Returns the text and the file it is for.
fn made(files: &[String]) -> Result<(String, &String), String> {
    match files {
        [flag, cards @ .., out] if flag == "--door-rows" && !cards.is_empty() && cards.len() <= 2 => {
            let host = card_of(&cards[0])?;
            let device = cards.get(1).map(|path| card_of(path)).transpose()?;
            Ok((door_rows_header(&host, device.as_ref())?, out))
        }
        [card, out] => Ok((facts_header(&card_of(card)?), out)),
        _ => Err("usage: gen_facts_header [--check] <card.json> <header> | [--check] --door-rows <host card> [<device card>] <header>".into()),
    }
}

/// Makes the header and writes it (or checks it).
fn run(args: &[String]) -> Result<(), String> {
    let (check, files) = match args {
        [flag, rest @ ..] if flag == "--check" => (true, rest),
        _ => (false, args),
    };
    let (text, header) = made(files)?;
    if check {
        let held = std::fs::read_to_string(header).map_err(|e| format!("{header}: {e}"))?;
        return if held == text {
            Ok(())
        } else {
            Err(format!("{header} differs from what its cards make"))
        };
    }
    std::fs::write(header, text).map_err(|e| format!("{header}: {e}"))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(why) => {
            eprintln!("gen_facts_header: {why}");
            ExitCode::FAILURE
        }
    }
}
