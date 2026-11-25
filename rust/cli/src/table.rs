pub fn print_table<S: AsRef<str>>(indent: &str, header: &[S], rows: &[Vec<String>]) {
    let mut widths = header
        .iter()
        .map(|cell| cell.as_ref().chars().count())
        .collect::<Vec<_>>();
    for row in rows {
        for (index, cell) in row.iter().enumerate() {
            let width = cell.chars().count();
            match widths.get_mut(index) {
                Some(current) => *current = (*current).max(width),
                None => widths.push(width),
            }
        }
    }
    let header = header
        .iter()
        .map(|cell| cell.as_ref().to_owned())
        .collect::<Vec<_>>();
    for row in std::iter::once(&header).chain(rows) {
        let line = row
            .iter()
            .enumerate()
            .map(|(index, cell)| {
                let pad = widths[index].saturating_sub(cell.chars().count());
                if index == 0 {
                    format!("{cell}{}", " ".repeat(pad))
                } else {
                    format!("{}{cell}", " ".repeat(pad))
                }
            })
            .collect::<Vec<_>>()
            .join("  ");
        println!("{indent}{}", line.trim_end());
    }
}
