//! The pixel seal, drawn with half-block characters so two rows of pixels fit in one line of text.

/// The seal of the favicon and the brand sheet, one string per row of pixels.
const SEAL: [&str; 11] = [
    ".......#####..............",
    ".....#########............",
    "##..###########...........",
    "###.###.####.###..........",
    ".###############..........",
    "..########.#####..........",
    "...##############........#",
    "...#################...###",
    "...######################.",
    "....####################..",
    ".....##################...",
];

/// The seal as lines of text, six of them, 26 columns wide.
pub fn seal() -> Vec<String> {
    let mut rows: Vec<&str> = SEAL.to_vec();
    let blank = ".".repeat(SEAL[0].len());
    if rows.len() % 2 == 1 {
        rows.push(&blank);
    }
    rows.chunks(2)
        .map(|pair| {
            pair[0]
                .bytes()
                .zip(pair[1].bytes())
                .map(|(top, bottom)| match (top == b'#', bottom == b'#') {
                    (true, true) => '█',
                    (true, false) => '▀',
                    (false, true) => '▄',
                    (false, false) => ' ',
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_seal_is_six_lines_of_equal_width() {
        let lines = seal();
        assert_eq!(lines.len(), 6);
        assert!(lines.iter().all(|l| l.chars().count() == 26));
    }

    #[test]
    fn every_row_of_pixels_has_the_same_width() {
        assert!(SEAL.iter().all(|r| r.len() == 26));
    }
}
