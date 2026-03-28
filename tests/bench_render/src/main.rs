use pdf_oxide::PdfDocument;
use pdf_oxide::rendering::{render_page, RenderOptions};
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench_render <pdf_path> [page_numbers...]");
        eprintln!("Example: bench_render attention.pdf 0 1 12 13 14");
        std::process::exit(1);
    }

    let pdf_path = &args[1];
    let mut doc = PdfDocument::open(pdf_path).expect("Failed to open PDF");
    let page_count = doc.page_count().expect("Failed to get page count");

    let pages: Vec<usize> = if args.len() > 2 {
        args[2..].iter().map(|s| s.parse::<usize>().unwrap()).collect()
    } else {
        (0..page_count).collect()
    };

    // Scale to match our pipeline: 2x scale ≈ 144 DPI (72 * 2)
    let opts = RenderOptions::with_dpi(144);

    println!("{:>6}  {:>10}  {:>10}  {:>12}", "Page", "Render", "Size", "Dims");
    println!("{}", "-".repeat(45));

    let mut total_time = 0.0f64;
    let mut slowest = (0usize, 0.0f64);

    for &pg in &pages {
        if pg >= page_count {
            eprintln!("Skipping page {} (only {} pages)", pg, page_count);
            continue;
        }

        let t0 = Instant::now();
        let image = render_page(&mut doc, pg, &opts).expect("Render failed");
        let elapsed = t0.elapsed().as_secs_f64();

        total_time += elapsed;
        if elapsed > slowest.1 {
            slowest = (pg, elapsed);
        }

        let flag = if elapsed > 0.5 {
            " <<< SLOW"
        } else if elapsed > 0.1 {
            " << moderate"
        } else {
            ""
        };

        println!(
            "{:>6}  {:>9.3}s  {:>9}B  {:>5}x{:<5}{}",
            pg,
            elapsed,
            image.data.len(),
            image.width,
            image.height,
            flag
        );
    }

    println!("{}", "-".repeat(45));
    println!(
        "Total: {:.3}s  Avg: {:.3}s  Slowest: page {} ({:.3}s)",
        total_time,
        total_time / pages.len() as f64,
        slowest.0,
        slowest.1
    );
}
