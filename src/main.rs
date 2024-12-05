use eframe::{egui, App};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use chrono::NaiveDate;
use plotters::prelude::*;
use std::path::Path;
use rand::prelude::*;
use rand::seq::SliceRandom;

#[derive(Debug, Clone)]
struct DrawEntry {
    date: NaiveDate,
    numbers: Vec<i32>,
    bonus_number: i32,
}

struct LottoApp {
    draws: Vec<DrawEntry>,
    set_size: usize,
    number_patterns: HashMap<i32, Vec<i32>>,
    predicted_numbers: Vec<i32>,
    predicted_bonus: i32,
    backtesting_results: Option<Vec<BacktestResult>>,
}

impl LottoApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let draws = Self::read_lotto_draws("649.csv");
        let mut app = Self {
            draws,
            set_size: 10,
            number_patterns: HashMap::new(),
            predicted_numbers: Vec::new(),
            predicted_bonus: 1,
            backtesting_results: None,
        };
        app.analyze_patterns();
        app
    }

    fn read_lotto_draws(file_path: &str) -> Vec<DrawEntry> {
        let file = match File::open(file_path) {
            Ok(file) => file,
            Err(err) => {
                eprintln!("Error opening file {}: {}", file_path, err);
                return Vec::new();
            }
        };
        let reader = BufReader::new(file);
        let mut draws = Vec::new();
        let mut line_number = 0;

        for line in reader.lines().skip(1) {
            line_number += 1;
            if let Ok(line) = line {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 10 {
                    let date_str = parts[3].trim_matches('"');
                    if let Ok(date) = NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
                        let mut numbers = Vec::new();
                        for i in 4..10 {
                            if let Ok(num) = parts[i].trim_matches('"').parse::<i32>() {
                                numbers.push(num);
                            }
                        }
                        if !numbers.is_empty() {
                            draws.push(DrawEntry { date, numbers, bonus_number: 1 });
                        } else {
                            eprintln!("No valid numbers found on line {}", line_number);
                        }
                    } else {
                        eprintln!("Invalid date format on line {}: {}", line_number, parts[3]);
                    }
                } else {
                    eprintln!("Invalid number of columns on line {}: expected >= 10, got {}", line_number, parts.len());
                }
            } else {
                eprintln!("Error reading line {}", line_number);
            }
        }

        if draws.is_empty() {
            eprintln!("Warning: No valid draws were read from the file");
        } else {
            println!("Successfully read {} draws", draws.len());
        }

        draws.sort_by(|a, b| a.date.cmp(&b.date));
        draws
    }

    fn analyze_patterns(&mut self) {
        let mut patterns: HashMap<i32, Vec<i32>> = HashMap::new();

        // Analyze each number's appearance pattern
        for num in 1..=49 {
            let mut appearances: Vec<usize> = Vec::new();
            for (i, draw) in self.draws.iter().enumerate() {
                if draw.numbers.contains(&num) {
                    appearances.push(i);
                }
            }

            // Calculate intervals between appearances
            if appearances.len() >= 2 {
                let intervals: Vec<i32> = appearances.windows(2)
                    .map(|w| (w[1] - w[0]) as i32)
                    .collect();
                patterns.insert(num, intervals);
            }
        }

        self.number_patterns = patterns;
        self.predict_next_numbers();
    }

    fn predict_next_numbers(&mut self) {
        let mut number_scores: HashMap<i32, f64> = HashMap::new();

        // Analyze each number's interval patterns
        for num in 1..=49 {
            let mut appearances: Vec<usize> = Vec::new();
            // Find all appearances of this number
            for (i, draw) in self.draws.iter().enumerate() {
                if draw.numbers.contains(&num) {
                    appearances.push(i);
                }
            }

            if appearances.len() >= 2 {
                // Calculate intervals between appearances
                let intervals: Vec<usize> = appearances.windows(2)
                    .map(|w| w[1] - w[0])
                    .collect();

                // Count frequency of each interval
                let mut interval_frequency: HashMap<usize, usize> = HashMap::new();
                for &interval in &intervals {
                    *interval_frequency.entry(interval).or_insert(0) += 1;
                }

                // Calculate score based on interval patterns
                let mut score = 0.0;

                if let Some(&last_appearance) = appearances.last() {
                    // Safely calculate draws since last appearance
                    let draws_since_last = self.draws.len().saturating_sub(last_appearance + 1);

                    // Calculate average interval and its frequency
                    let avg_interval: f64 = intervals.iter().sum::<usize>() as f64 / intervals.len() as f64;

                    // Find most common intervals (top 3)
                    let mut common_intervals: Vec<(usize, usize)> = interval_frequency.into_iter().collect();
                    common_intervals.sort_by(|a, b| b.1.cmp(&a.1));
                    common_intervals.truncate(3);

                    // Score based on matching patterns
                    for (interval, frequency) in common_intervals {
                        // Check if current gap matches this interval pattern
                        let interval_match = (draws_since_last as f64 - interval as f64).abs() <= 2.0;
                        if interval_match {
                            // Higher score for more frequent intervals
                            let pattern_score = (frequency as f64 / intervals.len() as f64) * 10.0;
                            score += pattern_score;

                            // Bonus for matching the most common interval
                            if frequency == intervals.len() {
                                score *= 1.5;
                            }
                        }
                    }

                    // Penalize recent appearances
                    if draws_since_last < 3 {
                        score *= 0.1; // Heavy penalty for very recent numbers
                    } else if draws_since_last < 5 {
                        score *= 0.5; // Moderate penalty for somewhat recent numbers
                    } else if draws_since_last > (avg_interval * 2.0) as usize {
                        score *= 1.2; // Bonus for numbers that haven't appeared in a while
                    }
                }

                if score > 0.0 {
                    number_scores.insert(num, score);
                }
            }
        }

        // Convert to vec and sort by score
        let mut scored_numbers: Vec<(i32, f64)> = number_scores.into_iter().collect();
        scored_numbers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // First, get all the highest scoring numbers (up to 49)
        let all_scored: Vec<i32> = scored_numbers.iter()
            .map(|(num, _)| *num)
            .collect();

        // Select top set_size numbers for the prediction set
        self.predicted_numbers = all_scored.iter()
            .take(self.set_size)
            .copied()
            .collect();

        // Sort the numbers for display
        self.predicted_numbers.sort_unstable();

        // Select the highest scoring number that isn't in our prediction set
        // Always look at the top 15 numbers for bonus selection, regardless of set_size
        self.predicted_bonus = all_scored.iter()
            .take(15) // Consider top 15 numbers for bonus
            .find(|&num| !self.predicted_numbers.contains(num))
            .copied()
            .unwrap_or(1);
    }

    fn predict_for_date(&self, historical_draws: &[DrawEntry], date: &NaiveDate) -> (Vec<i32>, i32) {
        let mut number_scores: HashMap<i32, f64> = HashMap::new();

        // Analyze each number's interval patterns
        for num in 1..=49 {
            let mut appearances: Vec<usize> = Vec::new();
            // Find all appearances of this number
            for (i, draw) in historical_draws.iter().enumerate() {
                if draw.numbers.contains(&num) {
                    appearances.push(i);
                }
            }

            if appearances.len() >= 2 {
                let intervals: Vec<usize> = appearances.windows(2)
                    .map(|w| w[1] - w[0])
                    .collect();

                let mut interval_frequency: HashMap<usize, usize> = HashMap::new();
                for &interval in &intervals {
                    *interval_frequency.entry(interval).or_insert(0) += 1;
                }

                let mut score = 0.0;

                if let Some(&last_appearance) = appearances.last() {
                    let draws_since_last = historical_draws.len().saturating_sub(last_appearance + 1);
                    let avg_interval: f64 = intervals.iter().sum::<usize>() as f64 / intervals.len() as f64;

                    // Calculate frequency score (0-1)
                    let frequency_score = appearances.len() as f64 / historical_draws.len() as f64;

                    // Calculate interval pattern score
                    let mut interval_pattern_score = 0.0;
                    let mut common_intervals: Vec<(usize, usize)> = interval_frequency.into_iter().collect();
                    common_intervals.sort_by(|a, b| b.1.cmp(&a.1));

                    for (interval, frequency) in common_intervals.iter().take(3) {
                        let interval_match_score = 1.0 / (1.0 + (draws_since_last as f64 - *interval as f64).abs());
                        let pattern_weight = *frequency as f64 / intervals.len() as f64;
                        interval_pattern_score += interval_match_score * pattern_weight;
                    }

                    // Calculate recency score (inverse of draws since last, normalized)
                    let recency_score = 1.0 / (1.0 + draws_since_last as f64 * 0.1);

                    // Weighted combination of all factors
                    let frequency_weight = 0.3;
                    let pattern_weight = 0.4;
                    let recency_weight = 0.3;

                    score = frequency_score * frequency_weight +
                           interval_pattern_score * pattern_weight +
                           recency_score * recency_weight;

                    // Adjusted penalties for recent appearances
                    if draws_since_last < 2 {
                        score *= 0.2; // Less severe penalty for very recent numbers
                    } else if draws_since_last < 4 {
                        score *= 0.7; // Moderate penalty for somewhat recent numbers
                    } else if draws_since_last > (avg_interval * 2.0) as usize {
                        score *= 1.3; // Increased bonus for overdue numbers
                    }

                    // Additional score for numbers that appear in common patterns
                    if common_intervals.len() >= 2 {
                        let pattern_consistency = common_intervals[0].1 as f64 / intervals.len() as f64;
                        if pattern_consistency > 0.3 {
                            score *= 1.2; // Bonus for consistent patterns
                        }
                    }
                }

                if score > 0.0 {
                    number_scores.insert(num, score);
                }
            }
        }

        // Convert to vec and sort by score
        let mut scored_numbers: Vec<(i32, f64)> = number_scores.into_iter().collect();
        scored_numbers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Get all scored numbers
        let all_scored: Vec<i32> = scored_numbers.iter()
            .map(|(num, _)| *num)
            .collect();

        // Select top set_size numbers for prediction
        let predicted_numbers: Vec<i32> = all_scored.iter()
            .take(self.set_size)
            .copied()
            .collect();

        // Select the highest scoring number that isn't in our prediction set
        // Always look at the top 15 numbers for bonus selection
        let predicted_bonus = all_scored.iter()
            .take(15) // Consider top 15 numbers for bonus
            .find(|&num| !predicted_numbers.contains(num))
            .copied()
            .unwrap_or(1);

        (predicted_numbers, predicted_bonus)
    }

    fn run_backtesting(&self, start_idx: usize) -> Vec<BacktestResult> {
        let mut results = Vec::new();

        for i in start_idx..self.draws.len() {
            let historical_draws: Vec<DrawEntry> = self.draws[0..i].to_vec();

            let (predicted_numbers, predicted_bonus) = self.predict_for_date(
                &historical_draws,
                &self.draws[i].date
            );

            let actual_numbers = self.draws[i].numbers.clone();
            let actual_bonus = self.draws[i].bonus_number;

            let matches = actual_numbers.iter()
                .filter(|&num| predicted_numbers.contains(num))
                .count();

            let bonus_matched = predicted_bonus == actual_bonus;

            results.push(BacktestResult {
                date: self.draws[i].date,
                matches,
                predicted_numbers,
                actual_numbers,
                bonus_matched,
            });
        }

        results
    }
}

#[derive(Debug)]
struct BacktestResult {
    date: NaiveDate,
    matches: usize,
    predicted_numbers: Vec<i32>,
    actual_numbers: Vec<i32>,
    bonus_matched: bool,
}

impl App for LottoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Lotto Number Analysis");

            // Display data status
            if self.draws.is_empty() {
                ui.colored_label(egui::Color32::RED, 
                    "No lottery data loaded. Please ensure '649.csv' exists in the application directory.");
                return;
            }

            ui.horizontal(|ui| {
                ui.label("Number set size:");
                if ui.add(egui::DragValue::new(&mut self.set_size).range(6..=49)).changed() {
                    self.predict_next_numbers();
                }
            });

            if let (Some(first), Some(last)) = (self.draws.first(), self.draws.last()) {
                ui.label(format!("Analyzing data from {} to {}", first.date, last.date));
            }

            if let Some(last_draw) = self.draws.last() {
                ui.heading(format!("Predictions for draws after {}", last_draw.date));
                ui.label("Predicted numbers for next draw:");
                ui.label(format!("Main numbers: {:?}", self.predicted_numbers));
                ui.label(format!("Bonus number: {}", self.predicted_bonus));

                ui.add_space(10.0);
                ui.label("(Based on frequency analysis and recent patterns)");
            }

            if ui.button("Run Backtesting with Visualization").clicked() {
                if !self.draws.is_empty() {
                    let start_idx = 10;
                    let results = self.run_backtesting(start_idx);
                    self.backtesting_results = Some(results);
                }
            }

            // Display backtesting results if available
            if let Some(ref results) = self.backtesting_results {
                ui.add_space(20.0);
                ui.heading("Backtesting Results");

                // Add scrollable area for metrics
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for result in results {
                        ui.label(format!("Date: {}", result.date));
                        ui.label(format!("Matches: {}", result.matches));
                        ui.label(format!("Predicted numbers: {:?}", result.predicted_numbers));
                        ui.label(format!("Actual numbers: {:?}", result.actual_numbers));
                        ui.label(format!("Bonus matched: {}", result.bonus_matched));
                        ui.add_space(10.0);
                    }
                });
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Lotto Analyzer",
        options,
        Box::new(|cc| Ok(Box::new(LottoApp::new(cc))))
    )
}
