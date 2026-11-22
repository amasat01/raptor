//! Migration of recorded IR documents from the old plane forms to the current one, with the same meaning.
//!
//! The stored recordings were made before a plane carried its axes and before the planes read per sample shared one role. The current loader accepts only the current form, and the
//! recorded unit texts are the reference of the writers, so the documents are rewritten here, in the one place every
//! replay tool reads recordings through, and nowhere else. The rules:
//! - a `PlaneExtent` of a plane becomes an `AxisExtent` of axis 0 (a plane without axes has one runtime axis);
//! - a read `At` (or a `TeamRead`, the read of a team result) keeps its index operand, with the clamp arithmetic it already holds, and states `edge = unchecked`, so that
//!   the new default `clamp` is not applied on top of it;
//! - a `WideWrite` or `AccumWrite` states `edge = unchecked`;
//! - a role `per_sample`, `vec_in` or `mat_in`, wherever a node, a slot or a wire names one, becomes `plane_in`: the value shape
//!   of the plane is already recorded beside it, and says what the three names said;
//! - no other field changes: the fields the new form added take their neutral values by being absent.
//!
//! The form is read from the node itself: a node of a class that always states `edge` in the current form, and does not, is
//! old; a `PlaneExtent` exists only in the old form. A node already in the current form is left as it is, so migrating twice,
//! or migrating a current document, changes nothing.
use rcore::json::J;
use std::collections::HashMap;

/// What one migration changed.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Changed {
    /// `PlaneExtent` nodes renamed to `AxisExtent`.
    pub extents: usize,
    /// Reads that gained `edge = unchecked`.
    pub reads: usize,
    /// Stores and accumulates that gained `edge = unchecked`.
    pub stores: usize,
    /// Roles renamed to `plane_in`.
    pub roles: usize,
}

impl Changed {
    /// Whether the document was already in the current form.
    pub fn is_none(&self) -> bool {
        *self == Changed::default()
    }

    /// Adds the counts of another migration.
    pub fn add(&mut self, other: Changed) {
        self.extents += other.extents;
        self.reads += other.reads;
        self.stores += other.stores;
        self.roles += other.roles;
    }
}

/// The tagged text `{"s": text}` the recordings use for a string value.
fn tagged(text: &str) -> J {
    J::Obj(vec![("s".to_string(), J::Str(text.to_string()))])
}

/// Sets `key` in the fields of a node, at its sorted position (the recordings write keys sorted) when it is new.
fn set_field(fields: &mut Vec<(String, J)>, key: &str, value: J) {
    match fields.iter().position(|(k, _)| k == key) {
        Some(at) => fields[at].1 = value,
        None => {
            let at = fields
                .iter()
                .position(|(k, _)| k.as_str() > key)
                .unwrap_or(fields.len());
            fields.insert(at, (key.to_string(), value));
        }
    }
}

/// The class name and the fields of a recorded node (`{"c": class, "m": module, "v": {fields}}`), when `doc` is one.
fn node_parts(doc: &mut J) -> Option<(&str, &mut Vec<(String, J)>)> {
    let J::Obj(entries) = doc else { return None };
    let mut class = None;
    let mut fields = None;
    for (key, value) in entries.iter_mut() {
        match (key.as_str(), value) {
            ("c", J::Str(name)) => class = Some(name.as_str()),
            ("v", J::Obj(inner)) => fields = Some(inner),
            _ => {}
        }
    }
    class.zip(fields)
}

/// The roles the planes read per sample had before they shared `plane_in`, picked by the rank of the value.
const OLD_READ_ROLES: [&str; 3] = ["per_sample", "vec_in", "mat_in"];

thread_local! {
    /// The old role each read plane the migration renamed carried, by name; the latest record migrated wins.
    static RECORDED: std::cell::RefCell<HashMap<String, String>> = std::cell::RefCell::default();
}

/// Forgets the old roles the previous record carried. The replay tools call it before each record they migrate.
pub fn forget_recorded_roles() {
    RECORDED.with(|state| state.borrow_mut().clear());
}

/// The first reason `doc` cannot be expressed in the one read role: two read planes of one name that carry different old
/// roles, which the single role, keyed by name, would merge into one slot.
pub fn role_conflict(doc: &J) -> Option<String> {
    fn collect(doc: &J, seen: &mut HashMap<String, String>) -> Option<String> {
        let pairs: Vec<(String, String)> = match doc {
            J::Obj(entries) => {
                entries
                    .iter()
                    .find_map(|(key, value)| match (key.as_str(), value) {
                        ("v", J::Obj(fields)) => {
                            let get = |field: &str| {
                                fields
                                    .iter()
                                    .find(|f| f.0 == field)
                                    .and_then(|f| tagged_text(&f.1))
                            };
                            get("role")
                                .zip(get("name"))
                                .map(|(role, name)| vec![(name.to_string(), role.to_string())])
                        }
                        ("t", J::Arr(pair)) if pair.len() == 2 => tagged_text(&pair[0])
                            .zip(tagged_text(&pair[1]))
                            .map(|(role, name)| vec![(name.to_string(), role.to_string())]),
                        _ => None,
                    })
            }
            _ => None,
        }
        .unwrap_or_default();
        for (name, role) in pairs
            .into_iter()
            .filter(|(_, r)| OLD_READ_ROLES.contains(&r.as_str()))
        {
            if let Some(first) = seen
                .insert(name.clone(), role.clone())
                .filter(|f| *f != role)
            {
                return Some(format!(
                    "two read planes named '{name}' carried the roles '{first}' and '{role}', which the one read role keyed by name merges into one slot"
                ));
            }
        }
        match doc {
            J::Obj(entries) => entries.iter().find_map(|(_, v)| collect(v, seen)),
            J::Arr(items) => items.iter().find_map(|v| collect(v, seen)),
            _ => None,
        }
    }
    collect(doc, &mut HashMap::new())
}

/// Notes that the read plane `name` carried the old role `old`.
fn note_recorded_role(name: &str, old: &str) {
    RECORDED.with(|state| state.borrow_mut().insert(name.to_string(), old.to_string()));
}

/// The role of a slot called `name` as a digest recorded before the planes read per sample shared one role took it: the old role
/// the recording named for that plane when the migration saw it, else the old name of the rank (`per_sample`, `vec_in`,
/// `mat_in`) for a `plane_in` of rank zero to two; every other role as it is.
pub fn old_role_name(role: &str, name: &str, rank: usize) -> String {
    if role != "plane_in" {
        return role.to_string();
    }
    let recorded = RECORDED.with(|state| state.borrow().get(name).cloned());
    recorded.unwrap_or_else(|| OLD_READ_ROLES.get(rank).map_or(role, |old| old).to_string())
}

/// A recorded refusal message in the current form: the two spellings of a role it names, `role 'NAME'` and `role='NAME'`, take
/// the current name when NAME is one of the old read roles. Nothing else in the message changes, so a message that only
/// contains the word `role` or an old name elsewhere is left as it is.
pub fn migrate_message(message: &str) -> String {
    let mut out = String::with_capacity(message.len());
    let mut rest = message;
    while let Some(at) = rest.find("role") {
        let (head, tail) = rest.split_at(at + 4);
        out.push_str(head);
        rest = tail;
        let opened = rest.strip_prefix(" '").or_else(|| rest.strip_prefix("='"));
        let Some(after) = opened else { continue };
        let quote = rest.len() - after.len();
        let Some(close) = after.find('\'') else {
            continue;
        };
        if OLD_READ_ROLES.contains(&&after[..close]) {
            out.push_str(&rest[..quote]);
            out.push_str("plane_in");
            rest = &after[close..];
        }
    }
    out.push_str(rest);
    out
}

/// Renames a tagged string `{"s": role}` to `plane_in` when it is one of the old names, and returns the old name.
fn rename_tagged(value: &mut J) -> Option<String> {
    let J::Obj(tagged) = value else { return None };
    match tagged.as_mut_slice() {
        [(tag, J::Str(role))] if tag == "s" && OLD_READ_ROLES.contains(&role.as_str()) => {
            Some(std::mem::replace(role, "plane_in".to_string()))
        }
        _ => None,
    }
}

/// The text of a tagged string `{"s": text}`.
fn tagged_text(value: &J) -> Option<&str> {
    match value {
        J::Obj(tagged) => match tagged.as_slice() {
            [(tag, J::Str(text))] if tag == "s" => Some(text),
            _ => None,
        },
        _ => None,
    }
}

/// Renames the old roles `doc` itself names: the `role` field of a record (a node, a slot, a wire), and the role of a
/// `(role, name)` pair, the form a slot list is written in.
fn rename_roles_of(doc: &mut J) -> usize {
    let J::Obj(entries) = doc else { return 0 };
    let mut renamed = 0;
    for (key, value) in entries.iter_mut() {
        match (key.as_str(), value) {
            ("v", J::Obj(fields)) => {
                let name = fields
                    .iter()
                    .find(|(field, _)| field == "name")
                    .and_then(|(_, value)| tagged_text(value))
                    .map(str::to_string);
                for (field, value) in fields.iter_mut() {
                    if let (true, Some(old)) = (field == "role", rename_tagged(value)) {
                        name.iter().for_each(|name| note_recorded_role(name, &old));
                        renamed += 1;
                    }
                }
            }
            ("t", J::Arr(pair))
                if pair.len() == 2
                    && matches!(&pair[1], J::Obj(n) if n.len() == 1 && n[0].0 == "s") =>
            {
                let name = tagged_text(&pair[1]).map(str::to_string);
                if let Some(old) = rename_tagged(&mut pair[0]) {
                    name.iter().for_each(|name| note_recorded_role(name, &old));
                    renamed += 1;
                }
            }
            _ => {}
        }
    }
    renamed
}

/// Migrates one node in place when it is in the old form; `Err` names a node the rules cannot express exactly.
fn migrate_node(doc: &mut J) -> Result<Changed, String> {
    let mut changed = Changed::default();
    let Some((class, fields)) = node_parts(doc) else {
        return Ok(changed);
    };
    let states_edge = fields.iter().any(|(k, _)| k == "edge");
    let operands = fields
        .iter()
        .find(|(k, _)| k == "operands")
        .and_then(|(_, o)| o.get("t"))
        .map_or(0, |o| o.arr().len());
    match class {
        "At" | "TeamRead" if !states_edge => {
            if operands != 2 {
                return Err(format!(
                    "an old read has the plane and one index, this one has {operands} operands"
                ));
            }
            set_field(fields, "edge", tagged("unchecked"));
            changed.reads = 1;
        }
        "WideWrite" | "AccumWrite" if !states_edge => {
            set_field(fields, "edge", tagged("unchecked"));
            changed.stores = 1;
        }
        "PlaneExtent" => {
            if operands != 1 {
                return Err(format!(
                    "a plane extent has the plane as its one operand, this one has {operands}"
                ));
            }
            set_field(
                fields,
                "axis",
                J::Obj(vec![("i".to_string(), J::Str("0".into()))]),
            );
            changed.extents = 1;
            let J::Obj(entries) = doc else {
                unreachable!("a node is an object")
            };
            for (key, value) in entries.iter_mut() {
                if key == "c" {
                    *value = J::Str("AxisExtent".into());
                }
            }
        }
        _ => {}
    }
    Ok(changed)
}

/// Renames the old read roles named anywhere in `doc` and changes nothing else: the form of a recording made after the planes
/// carried axes and before they shared one role. Returns how many roles it renamed.
pub fn migrate_roles(doc: &mut J) -> usize {
    let mut renamed = rename_roles_of(doc);
    match doc {
        J::Obj(entries) => entries
            .iter_mut()
            .for_each(|(_, v)| renamed += migrate_roles(v)),
        J::Arr(items) => items.iter_mut().for_each(|v| renamed += migrate_roles(v)),
        _ => {}
    }
    renamed
}

/// Migrates every node found anywhere in `doc` (a record, or the inputs of one) to the current form.
///
/// Returns what changed, or the reason of the first node that cannot be expressed exactly; the document is then left
/// partly migrated and must be discarded by the caller.
pub fn migrate(doc: &mut J) -> Result<Changed, String> {
    if let Some(why) = role_conflict(doc) {
        return Err(why);
    }
    migrate_tree(doc)
}

/// `migrate` over a document known to hold no role conflict, and over each part of it.
fn migrate_tree(doc: &mut J) -> Result<Changed, String> {
    let mut changed = migrate_node(doc)?;
    changed.roles = rename_roles_of(doc);
    match doc {
        J::Obj(entries) => {
            for (_, value) in entries.iter_mut() {
                changed.add(migrate_tree(value)?);
            }
        }
        J::Arr(items) => {
            for value in items.iter_mut() {
                changed.add(migrate_tree(value)?);
            }
        }
        _ => {}
    }
    Ok(changed)
}

/// The number tokens of a JSON text, as written, in order (a float written `2.0` stays distinct from the integer `2`).
fn number_tokens(source: &str) -> Vec<&str> {
    let (mut tokens, mut in_string, mut escaped, mut start) = (vec![], false, false, None);
    for (at, ch) in source.char_indices() {
        if in_string {
            (escaped, in_string) = (!escaped && ch == '\\', escaped || ch != '"');
            continue;
        }
        let numeric = ch.is_ascii_digit() || "+-.eE".contains(ch);
        match (start, numeric) {
            (None, true) if ch == '-' || ch.is_ascii_digit() => start = Some(at),
            (Some(from), false) => {
                tokens.push(&source[from..at]);
                start = None;
            }
            _ => {}
        }
        in_string = ch == '"';
    }
    tokens.extend(start.map(|from| &source[from..]));
    tokens
}

/// The text of `doc`, a migrated or parsed copy of the record written as `source`: compact, ASCII only, keys in the order held,
/// and every number as `source` wrote it. Migrating adds strings only, so the numbers keep their order.
pub fn render(doc: &J, source: &str) -> String {
    let mut out = String::new();
    write(doc, &mut number_tokens(source).into_iter(), &mut out);
    out
}

/// Appends the text of `doc` to `out`, taking each number from `numbers`.
fn write<'a>(doc: &J, numbers: &mut impl Iterator<Item = &'a str>, out: &mut String) {
    match doc {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Num(n) => out.push_str(
            numbers
                .next()
                .map_or_else(|| format!("{n:?}"), String::from)
                .as_str(),
        ),
        J::Str(s) => write_str(s, out),
        J::Arr(items) => {
            out.push('[');
            for (at, item) in items.iter().enumerate() {
                if at > 0 {
                    out.push(',');
                }
                write(item, numbers, out);
            }
            out.push(']');
        }
        J::Obj(entries) => {
            out.push('{');
            for (at, (key, value)) in entries.iter().enumerate() {
                if at > 0 {
                    out.push(',');
                }
                write_str(key, out);
                out.push(':');
                write(value, numbers, out);
            }
            out.push('}');
        }
    }
}

/// Appends a JSON string, escaping what the recorder escapes (quote, backslash, control and non-ASCII characters).
fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            _ => {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// The key line of a walk as it read before the plane-shape fields existed, when the node it describes states only the
/// neutral values of those fields; any other line is returned as it is.
///
/// The fields the move added take their neutral values in a migrated document: a read has no edge rule beyond the migrated
/// `unchecked`, a store states `unchecked` and no axes, an assignment no axes, and the extent of a flat plane asks for axis 0.
/// A line that states anything else does not match, so removing the fields can never hide a real difference.
pub fn remove_added_fields(line: &str) -> String {
    let line = old_read_role(line);
    let line = line.as_str();
    let (id, rest) = line.split_once('|').unwrap_or(("", line));
    let swap = |kind: &str, from: &str, to: &str| {
        rest.starts_with(kind)
            .then(|| rest.strip_suffix(from))
            .flatten()
            .map(|head| format!("{id}|{head}{to}"))
    };
    swap("('at',", ", ('None', 'unchecked'))", ", 'None')")
        .or_else(|| swap("('accum_write',", ", 'unchecked', ()))", "))"))
        .or_else(|| swap("('wide_write',", ", 'unchecked', ()))", "))"))
        .or_else(|| swap("('assign',", ", ()))", "))"))
        .or_else(|| {
            let head = rest.strip_prefix("('axis_extent', ")?;
            let head = head.strip_suffix(", 0)")?;
            Some(format!("{id}|('plane_extent', {head}, None)"))
        })
        .unwrap_or_else(|| line.to_string())
}

/// A numbered key line with the role a recording made before the planes read per sample shared one role gave its leaf; any other
/// line is returned as it is.
pub fn old_read_role(line: &str) -> String {
    let (id, rest) = line.split_once('|').unwrap_or(("", line));
    old_leaf_line(rest).map_or_else(|| line.to_string(), |old| format!("{id}|{old}"))
}

/// The key line of a `plane_in` leaf as it read with the old role of its rank, which the leaf's own shape gives; `None` for any
/// other line, and for a plane of rank three or more, which has no old form.
fn old_leaf_line(key: &str) -> Option<String> {
    let body = key
        .strip_prefix("('leaf', 'plane_in', ")?
        .strip_suffix(')')?;
    let (name, shape) = body.rsplit_once(", (")?;
    let rank = shape
        .trim_end_matches(')')
        .split(',')
        .filter(|e| !e.trim().is_empty())
        .count();
    let old = match RECORDED.with(|state| state.borrow().get(name.trim_matches('\'')).cloned()) {
        Some(recorded) => recorded,
        None => (*OLD_READ_ROLES.get(rank)?).to_string(),
    };
    Some(format!("('leaf', '{old}', {name})"))
}

/// What migrating a file of one record per line did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct FileReport {
    /// Records in the file.
    pub records: usize,
    /// Records rewritten (they held at least one old-form node).
    pub changed: usize,
    /// Records left as they were, each with the reason.
    pub refused: Vec<(usize, String)>,
}

/// Migrates a file of one JSON record per line in place; a record already in the current form is kept byte for byte.
///
/// A record is rewritten only when writing its unmigrated parse back reproduces its line exactly (so that nothing but the
/// migration changes); otherwise it is kept and listed with the reason.
pub fn migrate_file(path: &std::path::Path) -> Result<FileReport, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut report = FileReport::default();
    let mut out = String::with_capacity(text.len());
    for (at, line) in text.lines().enumerate() {
        report.records += 1;
        let kept = match rewrite_line(line) {
            Ok(Some(new)) => {
                report.changed += 1;
                new
            }
            Ok(None) => line.to_string(),
            Err(reason) => {
                report.refused.push((at + 1, reason));
                line.to_string()
            }
        };
        out.push_str(&kept);
        out.push('\n');
    }
    std::fs::write(path, out).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(report)
}

/// The migrated line, or `None` when the record holds nothing old; `Err` when it cannot be rewritten exactly.
fn rewrite_line(line: &str) -> Result<Option<String>, String> {
    let mut record = rcore::json::parse(line)?;
    if render(&record, line) != line {
        return Err("writing the record back does not reproduce its line".into());
    }
    if migrate(&mut record)?.is_none() {
        return Ok(None);
    }
    Ok(Some(render(&record, line)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rcore::json::parse;

    fn doc(text: &str) -> J {
        parse(text).expect("the test document is JSON")
    }

    const READ: &str = r#"{"c":"At","m":"raptor.ir.nodes","v":{"kind":{"s":"at"},"literal":null,"operands":{"t":[{"n":1},{"n":4}]}}}"#;
    const STORE: &str = r#"{"c":"AccumWrite","m":"raptor.ir.nodes","v":{"contract":{"s":"auto"},"operands":{"t":[{"n":0},{"n":3}]}}}"#;
    const EXTENT: &str =
        r#"{"c":"PlaneExtent","m":"raptor.ir.nodes","v":{"operands":{"t":[{"n":2}]}}}"#;

    #[test]
    fn the_neutral_added_fields_are_removed_from_key_lines_and_only_those() {
        let cases = [
            ("5|('at', 'f64', (), None, (1, 4), ('None', 'unchecked'))", "5|('at', 'f64', (), None, (1, 4), 'None')"),
            ("4|('accum_write', 'f64', (), None, (0, 3), ('accum_out', 'cut0', 'unchecked', ()))", "4|('accum_write', 'f64', (), None, (0, 3), ('accum_out', 'cut0'))"),
            ("5|('assign', 'f64', (), None, (4,), ('mutable', 'y', ()))", "5|('assign', 'f64', (), None, (4,), ('mutable', 'y'))"),
            ("7|('axis_extent', 'p', (), None, (3,), 0)", "7|('plane_extent', 'p', (), None, (3,), None)"),
            // a read that states a real edge rule, a store with axes and an extent of axis 1 are not neutral: kept
            ("5|('at', 'f64', (), None, (1, 4), ('None', 'clamp'))", "5|('at', 'f64', (), None, (1, 4), ('None', 'clamp'))"),
            ("5|('assign', 'f64', (), None, (4,), ('mutable', 'y', (('i', 'n'),)))", "5|('assign', 'f64', (), None, (4,), ('mutable', 'y', (('i', 'n'),)))"),
            ("7|('axis_extent', 'p', (), None, (3,), 1)", "7|('axis_extent', 'p', (), None, (3,), 1)"),
            ("0|('leaf', 'uniform', 'a')", "0|('leaf', 'uniform', 'a')"),
            // a read plane's key states its value shape; the old line names the role of that rank, and rank three has none
            ("0|('leaf', 'plane_in', 'x', ())", "0|('leaf', 'per_sample', 'x')"),
            ("0|('leaf', 'plane_in', 'x', (3,))", "0|('leaf', 'vec_in', 'x')"),
            ("0|('leaf', 'plane_in', 'm, (1)', (2, 3))", "0|('leaf', 'mat_in', 'm, (1)')"),
            ("0|('leaf', 'plane_in', 't', (2, 3, 4))", "0|('leaf', 'plane_in', 't', (2, 3, 4))"),
        ];
        for (current, old) in cases {
            assert_eq!(remove_added_fields(current), old);
        }
    }

    #[test]
    fn the_three_old_read_roles_become_plane_in_in_every_node_that_names_one() {
        let leaf = |role: &str| {
            format!(
                r#"{{"c":"Leaf","m":"raptor.ir.nodes","v":{{"kind":{{"s":"vocab_read"}},"role":{{"s":"{role}"}},"name":{{"s":"x"}}}}}}"#
            )
        };
        for old in ["per_sample", "vec_in", "mat_in"] {
            let mut node = doc(&leaf(old));
            let changed = migrate(&mut node).unwrap();
            assert_eq!(
                changed,
                Changed {
                    roles: 1,
                    ..Changed::default()
                }
            );
            assert_eq!(render(&node, &leaf(old)), leaf("plane_in"));
        }
        // a role that is not one of the three, and a node already in the current form, are left alone
        for kept in ["lookup", "mutable", "plane_in"] {
            let mut node = doc(&leaf(kept));
            assert!(migrate(&mut node).unwrap().is_none());
        }
        // a slot or a wire inside a larger document is reached as well
        let mut slot = doc(
            r#"{"declared":{"l":[{"c":"Slot","m":"raptor.ir.nodes","v":{"role":{"s":"mat_in"},"name":{"s":"m"}}}]}}"#,
        );
        assert_eq!(migrate(&mut slot).unwrap().roles, 1);
    }

    #[test]
    fn the_old_name_of_a_role_follows_the_rank_of_its_value() {
        forget_recorded_roles();
        assert_eq!(old_role_name("plane_in", "x", 0), "per_sample");
        assert_eq!(old_role_name("plane_in", "x", 1), "vec_in");
        assert_eq!(old_role_name("plane_in", "x", 2), "mat_in");
        assert_eq!(old_role_name("plane_in", "x", 3), "plane_in");
        assert_eq!(old_role_name("mutable", "x", 1), "mutable");
    }

    #[test]
    fn numbers_keep_the_form_they_were_written_in() {
        let source = r#"{"a":[2.0,2,-1.5e-05,"1.0e3"],"b":true,"c":null}"#;
        assert_eq!(render(&doc(source), source), source);
    }

    #[test]
    fn a_plane_extent_becomes_the_extent_of_axis_zero() {
        let mut node = doc(EXTENT);
        let changed = migrate(&mut node).unwrap();
        assert_eq!(
            changed,
            Changed {
                extents: 1,
                ..Changed::default()
            }
        );
        assert_eq!(
            render(&node, EXTENT),
            r#"{"c":"AxisExtent","m":"raptor.ir.nodes","v":{"axis":{"i":"0"},"operands":{"t":[{"n":2}]}}}"#
        );
    }

    #[test]
    fn an_old_read_keeps_its_index_and_states_no_edge_rule() {
        let mut node = doc(READ);
        let changed = migrate(&mut node).unwrap();
        assert_eq!(
            changed,
            Changed {
                reads: 1,
                ..Changed::default()
            }
        );
        assert_eq!(
            render(&node, READ),
            r#"{"c":"At","m":"raptor.ir.nodes","v":{"edge":{"s":"unchecked"},"kind":{"s":"at"},"literal":null,"operands":{"t":[{"n":1},{"n":4}]}}}"#
        );
    }

    #[test]
    fn an_old_store_states_no_edge_rule_and_gains_no_axes() {
        let mut node = doc(STORE);
        let changed = migrate(&mut node).unwrap();
        assert_eq!(
            changed,
            Changed {
                stores: 1,
                ..Changed::default()
            }
        );
        let text = render(&node, STORE);
        assert!(text.contains(r#""edge":{"s":"unchecked"}"#), "{text}");
        assert!(!text.contains("axes"), "{text}");
    }

    #[test]
    fn an_old_team_read_states_no_edge_rule_like_an_old_read() {
        let mut node = doc(
            r#"{"c":"TeamRead","m":"raptor.ir.nodes","v":{"kind":{"s":"at"},"operands":{"t":[{"n":1},{"n":4}]}}}"#,
        );
        let changed = migrate(&mut node).unwrap();
        assert_eq!(changed.reads, 1);
        assert!(render(&node, "").contains(r#""edge":{"s":"unchecked"}"#));
    }

    #[test]
    fn a_read_with_two_indices_in_the_old_form_is_refused_by_name() {
        let mut node = doc(r#"{"c":"At","v":{"operands":{"t":[{"n":0},{"n":1},{"n":2}]}}}"#);
        let reason = migrate(&mut node).unwrap_err();
        assert!(reason.contains("3 operands"), "{reason}");
    }

    #[test]
    fn nodes_inside_a_record_are_found_and_counted() {
        let mut record = doc(&format!(
            r#"{{"inputs":{{"nodes":[{READ},{STORE},{EXTENT}]}}}}"#
        ));
        let changed = migrate(&mut record).unwrap();
        assert_eq!(
            changed,
            Changed {
                extents: 1,
                reads: 1,
                stores: 1,
                ..Changed::default()
            }
        );
    }

    #[test]
    fn a_current_document_is_left_as_it_is() {
        let source = format!(r#"[{READ},{STORE},{EXTENT}]"#);
        let mut once = doc(&source);
        migrate(&mut once).unwrap();
        let text = render(&once, &source);
        let mut twice = doc(&text);
        assert!(migrate(&mut twice).unwrap().is_none());
        assert_eq!(render(&twice, &text), text);
    }

    #[test]
    fn a_migrated_document_loads_in_the_current_ir() {
        use crate::corpus::{declare, make_nodes, manifests, Vals};
        use raptor_ir::door::rx_ctx_new;
        let leaf = |name: &str, role: &str, kind: &str, dtype: &str| {
            format!(
                r#"{{"c":"Leaf","m":"raptor.ir.nodes","v":{{"kind":{{"s":"{kind}"}},"name":{{"s":"{name}"}},"operands":{{"t":[]}},"quantity":null,"role":{{"s":"{role}"}},"sorted":false,"ttype":{{"m":"raptor.ir.types","r":"TensorType","v":{{"dtype":{{"s":"{dtype}"}},"shape":{{"t":[]}},"tag":null}}}},"wire_index":{{"i":"0"}}}}}}"#
            )
        };
        let ttype = |dtype: &str| {
            format!(
                r#"{{"m":"raptor.ir.types","r":"TensorType","v":{{"dtype":{{"s":"{dtype}"}},"shape":{{"t":[]}},"tag":null}}}}"#
            )
        };
        // a lookup plane, an index leaf, the old extent of the plane and an old read of it at the index
        let nodes = [
            leaf("t", "lookup", "table_read", "f64"),
            leaf("i", "per_sample", "vocab_read", "i32"),
            format!(
                r#"{{"c":"PlaneExtent","m":"raptor.ir.nodes","v":{{"operands":{{"t":[{{"n":0}}]}},"ttype":{}}}}}"#,
                ttype("i32")
            ),
            format!(
                r#"{{"c":"At","m":"raptor.ir.nodes","v":{{"kind":{{"s":"at"}},"literal":null,"operands":{{"t":[{{"n":0}},{{"n":1}}]}},"ttype":{}}}}}"#,
                ttype("f64")
            ),
        ];
        let mut inputs = doc(&format!(
            r#"{{"kinds":[],"nodes":[{}],"objs":[],"options":{{}}}}"#,
            nodes.join(",")
        ));
        assert_eq!(
            migrate(&mut inputs).unwrap(),
            Changed {
                extents: 1,
                reads: 1,
                roles: 1,
                ..Changed::default()
            }
        );
        let ctx = rx_ctx_new();
        unsafe { raptor_wings::declare::rw_declare_kinds(ctx) };
        declare(ctx, &manifests(&inputs)).unwrap();
        let handles = make_nodes(ctx, &inputs, &mut Vals::default()).unwrap();
        assert_eq!(handles.len(), 4);
    }

    #[test]
    fn a_file_is_migrated_in_place_and_counted() {
        let dir = std::env::temp_dir().join(format!("migrate_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("records.jsonl");
        let old = format!(r#"{{"inputs":{{"nodes":[{READ}]}}}}"#);
        let current = r#"{"inputs":{"nodes":[]}}"#;
        std::fs::write(&path, format!("{old}\n{current}\n")).unwrap();
        let report = migrate_file(&path).unwrap();
        assert_eq!(
            (report.records, report.changed, report.refused.len()),
            (2, 1, 0)
        );
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text
            .lines()
            .next()
            .unwrap()
            .contains(r#""edge":{"s":"unchecked"}"#));
        assert_eq!(text.lines().nth(1).unwrap(), current);
        let again = migrate_file(&path).unwrap();
        assert_eq!(again.changed, 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Oracle: the two spellings the tools print a role in are renamed; the word alone and other quoted names are not.
    #[test]
    fn a_recorded_message_names_the_current_role_in_its_two_role_forms_only() {
        assert_eq!(
            migrate_message("wire 'q' (role 'vec_in') is bound with no owning quantity"),
            "wire 'q' (role 'plane_in') is bound with no owning quantity"
        );
        assert_eq!(
            migrate_message(
                "collision on (role='per_sample', name='x') and (role='mat_in', name='y')"
            ),
            "collision on (role='plane_in', name='x') and (role='plane_in', name='y')"
        );
        let unchanged =
            "the role of 'vec_in' is a name, role 'mutable' stays, a role is here, name='vec_in'";
        assert_eq!(migrate_message(unchanged), unchanged);
    }

    /// Oracle: the recording's own old role is carried, not derived from the rank (a rank-0 plane that was `vec_in`), and two old
    /// roles under one name cannot be expressed.
    #[test]
    fn the_old_role_of_a_plane_is_the_recorded_one_and_two_under_one_name_cannot_be_expressed() {
        use rcore::json::parse;
        forget_recorded_roles();
        let leaf = |role: &str| {
            format!(r#"{{"c":"Leaf","m":"m","v":{{"name":{{"s":"x"}},"role":{{"s":"{role}"}}}}}}"#)
        };
        let mut one = parse(&leaf("vec_in")).unwrap();
        assert!(migrate(&mut one).is_ok());
        assert_eq!(old_role_name("plane_in", "x", 0), "vec_in");
        let two = parse(&format!(
            r#"{{"l":[{},{}]}}"#,
            leaf("vec_in"),
            leaf("per_sample")
        ))
        .unwrap();
        let why = role_conflict(&two).unwrap();
        assert!(
            why.contains("two read planes named 'x'") && why.contains("'vec_in' and 'per_sample'"),
            "{why}"
        );
        assert_eq!(
            role_conflict(
                &parse(&format!(
                    r#"{{"l":[{},{}]}}"#,
                    leaf("vec_in"),
                    leaf("vec_in")
                ))
                .unwrap()
            ),
            None
        );
    }
}
