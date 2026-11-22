//! Migration of recorded IR documents from the old plane forms to the current one, with the same meaning.
//!
//! The stored recordings were made before a plane carried its axes and before the planes read per sample shared one role. The current loader accepts only the current form, and the
//! recorded unit texts are the reference of the writers, so the documents are rewritten here, in the one place every
//! replay tool reads recordings through, and nowhere else. The rules:
//! - a `PlaneExtent` of a plane becomes an `AxisExtent` of axis 0 (a plane without axes has one runtime axis);
//! - a read `At` (or a `TeamRead`, the read of a team result) keeps its index operand, with the clamp arithmetic it already holds, and states `edge = unchecked`, so that
//!   the new default `clamp` is not applied on top of it;
//! - a `WideWrite` states `edge = unchecked` and becomes an `Update` of the operation `store`; an `AccumWrite` states
//!   `edge = unchecked` and becomes an `Update` of the operation `add` (`or` for the contract `or`) whose summation and order
//!   are what its contract meant: `plain` is the plain summation, `compensated` and `keyed_auto` the compensated one,
//!   `ordered` the order `lane`, `auto` the neutral values;
//! - a role `per_sample`, `vec_in` or `mat_in`, wherever a node, a slot or a wire names one, becomes `plane_in`: the value shape
//!   of the plane is already recorded beside it, and says what the three names said;
//! - a role `wide_out` or `accum_out` becomes `plane_out`: whether the runtime zeroes the plane follows from the updates that
//!   write it, so the old name is only kept, by plane name, for the digests recorded with it;
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
    /// Stores and accumulates that became updates.
    pub stores: usize,
    /// Roles renamed to `plane_in` or `plane_out`.
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

/// The roles the written planes had before they shared `plane_out`.
const OLD_WRITE_ROLES: [&str; 2] = ["wide_out", "accum_out"];

/// The role of a written plane no recording named, where the key line states no operation: the accumulates are the planes the
/// wings build most.
const DEFAULT_OLD_WRITE_ROLE: &str = "accum_out";

/// The current role an old role name became, when it is one.
fn current_role(old: &str) -> Option<&'static str> {
    if OLD_READ_ROLES.contains(&old) {
        Some("plane_in")
    } else if OLD_WRITE_ROLES.contains(&old) {
        Some("plane_out")
    } else {
        None
    }
}

thread_local! {
    /// The old role each read plane the migration renamed carried, by name; the latest record migrated wins.
    static RECORDED: std::cell::RefCell<HashMap<String, String>> = std::cell::RefCell::default();
    /// The contract each accumulate plane the migration turned into an update named, by plane name, where the current form
    /// keeps less than the contract said (`keyed_auto` and `compensated` are one summation); the latest record wins.
    static RECORDED_CONTRACTS: std::cell::RefCell<HashMap<String, String>> = std::cell::RefCell::default();
}

/// Forgets the old roles the previous record carried. The replay tools call it before each record they migrate.
pub fn forget_recorded_roles() {
    RECORDED.with(|state| state.borrow_mut().clear());
    RECORDED_CONTRACTS.with(|state| state.borrow_mut().clear());
}

/// The first reason `doc` cannot be expressed in the shared roles: two planes of one name that carry different old roles of
/// one family (the read roles, or the written ones), which the single role, keyed by name, would merge into one slot.
pub fn role_conflict(doc: &J) -> Option<String> {
    fn collect(doc: &J, seen: &mut HashMap<(&'static str, String), String>) -> Option<String> {
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
        for (name, role) in pairs {
            let Some(current) = current_role(&role) else {
                continue;
            };
            if let Some(first) = seen
                .insert((current, name.clone()), role.clone())
                .filter(|f| *f != role)
            {
                return Some(format!(
                    "two planes named '{name}' carried the roles '{first}' and '{role}', which the one role '{current}' keyed by name merges into one slot"
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

/// The role of a slot called `name` as a digest recorded before the planes shared roles took it: the old role the recording
/// named for that plane when the migration saw it, else the old name of the rank (`per_sample`, `vec_in`, `mat_in`) for a
/// `plane_in` of rank zero to two, and for a `plane_out` the role its zeroing says (`accum_out` when the runtime zeroes it,
/// else `wide_out`); every other role as it is.
pub fn old_role_name(role: &str, name: &str, rank: usize, zeroed: bool) -> String {
    if role != "plane_in" && role != "plane_out" {
        return role.to_string();
    }
    let recorded = RECORDED.with(|state| state.borrow().get(name).cloned());
    recorded.unwrap_or_else(|| match role {
        "plane_out" if zeroed => "accum_out".to_string(),
        "plane_out" => "wide_out".to_string(),
        _ => OLD_READ_ROLES.get(rank).map_or(role, |old| old).to_string(),
    })
}

/// The list of sink roles a recorded refusal states, in the old and the current form.
const OLD_SINK_ROLE_LIST: &str = "('mutable', 'out', 'wide_out', 'accum_out')";
const SINK_ROLE_LIST: &str = "('mutable', 'out', 'plane_out')";

/// A recorded refusal message in the current form: the two spellings of a role it names, `role 'NAME'` and `role='NAME'`, take
/// the current name when NAME is one of the old roles, and the list of sink roles it states is the current list. Nothing else in
/// the message changes, so a message that only contains the word `role` or an old name elsewhere is left as it is.
pub fn migrate_message(message: &str) -> String {
    let message = message.replace(OLD_SINK_ROLE_LIST, SINK_ROLE_LIST);
    let message = message.as_str();
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
        if let Some(current) = current_role(&after[..close]) {
            out.push_str(&rest[..quote]);
            out.push_str(current);
            rest = &after[close..];
        }
    }
    out.push_str(rest);
    out
}

/// Renames a tagged string `{"s": role}` to the current role when it is an old name, and returns the old name.
fn rename_tagged(value: &mut J) -> Option<String> {
    let J::Obj(tagged) = value else { return None };
    match tagged.as_mut_slice() {
        [(tag, J::Str(role))] if tag == "s" && current_role(role).is_some() => {
            let current = current_role(role).map(str::to_string)?;
            Some(std::mem::replace(role, current))
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

/// Renames the class a recorded dialect manifest says a kind refines, and the wire name it states: a kind that refined
/// `wide_write` refines `update`, which is a store when it states no operation.
fn rename_refined_class(doc: &mut J) {
    let J::Obj(entries) = doc else { return };
    for (key, value) in entries.iter_mut() {
        if matches!(key.as_str(), "refines" | "wire") && tagged_text(value) == Some("wide_write") {
            *value = tagged("update");
        }
    }
}

/// Renames the `kind` a recorded node states when it is the wire name of an old write class: a node of a wing kind that
/// refines the store stated the wire name of the class it refined.
fn rename_node_kind(doc: &mut J) {
    let Some((_, fields)) = node_parts(doc) else {
        return;
    };
    for (key, value) in fields.iter_mut() {
        if key == "kind" && matches!(tagged_text(value), Some("wide_write" | "accum_write")) {
            *value = tagged("update");
        }
    }
}

/// Gives the recorded node `doc` the class `class`.
fn rename_class(doc: &mut J, class: &str) {
    let J::Obj(entries) = doc else {
        unreachable!("a node is an object")
    };
    for (key, value) in entries.iter_mut() {
        if key == "c" {
            *value = J::Str(class.into());
        }
    }
}

/// The operation, summation and order the update that replaces an old write states: a store, or an accumulate with the
/// contract it recorded (`or` is the bit-or operation; the contracts `compensated` and `keyed_auto` are the compensated
/// summation, `plain` the plain one, `ordered` the order `lane`).
pub fn update_fields(
    accumulates: bool,
    contract: &str,
) -> (&'static str, &'static str, &'static str) {
    match (accumulates, contract) {
        (false, _) => ("store", "auto", "any"),
        (true, "or") => ("or", "auto", "any"),
        (true, "plain") => ("add", "plain", "any"),
        (true, "compensated" | "keyed_auto") => ("add", "compensated", "any"),
        (true, "ordered") => ("add", "auto", "lane"),
        (true, _) => ("add", "auto", "any"),
    }
}

/// Turns the recorded `WideWrite` or `AccumWrite` `doc` into the `Update` that means the same, with the contract it recorded
/// restated as the operation, summation and order (`restate_edge` also gives it the migrated `edge = unchecked`, for a
/// recording that predates the edge rule). Returns whether `doc` was one of the two old classes.
fn become_update(doc: &mut J, restate_edge: bool) -> bool {
    let Some((class, fields)) = node_parts(doc) else {
        return false;
    };
    let accumulates = match class {
        "WideWrite" => false,
        "AccumWrite" => true,
        _ => return false,
    };
    if restate_edge && !fields.iter().any(|(k, _)| k == "edge") {
        set_field(fields, "edge", tagged("unchecked"));
    }
    let contract = fields
        .iter()
        .position(|(k, _)| k == "contract")
        .map(|at| fields.remove(at).1);
    let contract = contract.as_ref().and_then(tagged_text).unwrap_or("auto");
    let (op, summation, order) = update_fields(accumulates, contract);
    let name = fields
        .iter()
        .find(|(k, _)| k == "name")
        .and_then(|(_, v)| tagged_text(v))
        .map(str::to_string);
    if let Some(name) = name.filter(|_| contract == "keyed_auto") {
        note_recorded_contract(&name, contract);
    }
    set_field(fields, "op", tagged(op));
    set_field(fields, "summation", tagged(summation));
    set_field(fields, "order", tagged(order));
    if fields.iter().any(|(k, _)| k == "kind") {
        set_field(fields, "kind", tagged("update"));
    }
    rename_class(doc, "Update");
    true
}

/// States `zeroed = true` in a recorded declared `Slot` whose role is `accum_out`: the current form keeps one written-plane role,
/// and says by this member which of its planes the runtime zeroes. Returns whether it did.
fn state_zeroing(doc: &mut J) -> bool {
    let J::Obj(entries) = doc else { return false };
    let is_slot = entries
        .iter()
        .any(|(k, v)| k == "r" && matches!(v, J::Str(class) if class == "Slot"));
    if !is_slot {
        return false;
    }
    for (key, value) in entries.iter_mut() {
        let J::Obj(fields) = value else { continue };
        let accumulated = key == "v"
            && fields
                .iter()
                .any(|(k, v)| k == "role" && tagged_text(v) == Some("accum_out"));
        if accumulated {
            set_field(fields, "zeroed", J::Bool(true));
            return true;
        }
    }
    false
}

/// States, in a recorded walk (a record with an `arg_spec`), the names of its planes that had the role `accum_out` as
/// `zeroed_outputs`: the current form keeps one written-plane role, and says by this member which of its planes the runtime
/// zeroes. Returns whether it did.
fn state_zeroed_outputs(doc: &mut J) -> bool {
    let J::Obj(entries) = doc else { return false };
    let zeroed: Vec<J> = entries
        .iter()
        .filter(|(key, _)| key == "arg_spec")
        .filter_map(|(_, spec)| spec.get("t").or(spec.get("l")))
        .flat_map(|pairs| pairs.arr().iter())
        .filter_map(|pair| pair.get("t"))
        .filter(|pair| pair.arr().len() == 2 && tagged_text(&pair.arr()[0]) == Some("accum_out"))
        .map(|pair| pair.arr()[1].clone())
        .collect();
    if zeroed.is_empty() {
        return false;
    }
    let at = entries
        .iter()
        .position(|(k, _)| k.as_str() > "zeroed_outputs")
        .unwrap_or(entries.len());
    entries.insert(
        at,
        (
            "zeroed_outputs".into(),
            J::Obj(vec![("t".into(), J::Arr(zeroed))]),
        ),
    );
    true
}

/// Notes that the accumulate plane `name` carried the old contract `contract`, which the current form keeps as a summation only.
pub fn note_recorded_contract(name: &str, contract: &str) {
    RECORDED_CONTRACTS.with(|state| {
        state
            .borrow_mut()
            .insert(name.to_string(), contract.to_string())
    });
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
        "WideWrite" | "AccumWrite" => {
            changed.stores = usize::from(become_update(doc, true));
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
            rename_class(doc, "AxisExtent");
        }
        _ => {}
    }
    Ok(changed)
}

/// Renames the old roles named anywhere in `doc` and turns the old write classes into updates, and changes nothing else: the
/// form of a recording made after the planes carried axes and before they shared roles. Returns how many it renamed.
pub fn migrate_roles(doc: &mut J) -> usize {
    rename_refined_class(doc);
    rename_node_kind(doc);
    let mut renamed = usize::from(become_update(doc, false));
    state_zeroing(doc);
    state_zeroed_outputs(doc);
    renamed += rename_roles_of(doc);
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
    rename_refined_class(doc);
    rename_node_kind(doc);
    let mut changed = migrate_node(doc)?;
    state_zeroing(doc);
    state_zeroed_outputs(doc);
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
    old_leaf_line(rest)
        .or_else(|| old_update_line(rest))
        .or_else(|| old_write_role(rest))
        .map(|old| old_body_sinks(&old))
        .or_else(|| {
            rest.contains("('update', 'plane_out', '")
                .then(|| old_body_sinks(rest))
        })
        .map_or_else(|| line.to_string(), |old| format!("{id}|{old}"))
}

/// The key line of a sink of any class as it read with the old role of the plane it writes, which the recordings named by plane
/// name; `None` for a line whose tail names no `plane_out` plane. A kind that refines the update and states its own tail still
/// reads `update` at its head: it is given the class that wrote the plane, as the recording named it.
fn old_write_role(key: &str) -> Option<String> {
    let (head, tail) = key.split_once(", ('plane_out', '")?;
    let plane = &tail[..tail.find('\'')?];
    let role = recorded_write_role(plane, DEFAULT_OLD_WRITE_ROLE);
    let head = head.strip_prefix("('update', ").map_or_else(
        || head.to_string(),
        |rest| format!("('{}', {rest}", old_write_kind(&role)),
    );
    Some(format!("{head}, ('{role}', '{tail}"))
}

/// The old role the recordings named for the written plane `plane`, else `unrecorded`.
fn recorded_write_role(plane: &str, unrecorded: &str) -> String {
    RECORDED
        .with(|state| state.borrow().get(plane).cloned())
        .unwrap_or_else(|| unrecorded.to_string())
}

/// The old write kind of a plane with the old role `role`: the class that wrote it.
fn old_write_kind(role: &str) -> &'static str {
    if role == "wide_out" {
        "wide_write"
    } else {
        "accum_write"
    }
}

/// A loop key line states each body sink as `(kind, role, name)`: those of the current form as the old kind and role.
fn old_body_sinks(line: &str) -> String {
    const OPEN: &str = "('update', 'plane_out', '";
    let mut out = String::new();
    let mut rest = line;
    while let Some(at) = rest.find(OPEN) {
        let after = &rest[at + OPEN.len()..];
        let Some(close) = after.find('\'') else { break };
        let role = recorded_write_role(&after[..close], DEFAULT_OLD_WRITE_ROLE);
        out.push_str(&rest[..at]);
        out.push_str(&format!(
            "('{}', '{role}', '{}",
            old_write_kind(&role),
            &after[..close]
        ));
        rest = &after[close..];
    }
    out.push_str(rest);
    out
}

/// The key line of an `update` as it read as a `wide_write` or an `accum_write`: the old class, the old role of its plane and, for
/// an accumulate, the contract the fields say; `None` for any other line. The current tail is
/// `(role, name, op, conflict, [compensated,] order, returns_old, edge, axes)`.
fn old_update_line(key: &str) -> Option<String> {
    let body = key.strip_prefix("('update', ")?;
    let (head, tail) = body.split_once(", ('plane_out', ")?;
    let mut parts = tail.splitn(4, ", ");
    let (name, op, _conflict) = (parts.next()?, parts.next()?, parts.next()?);
    let rest = parts.next()?;
    let (summation, rest) = match rest.strip_prefix("'compensated', ") {
        Some(after) => ("compensated", after),
        None => ("auto", rest),
    };
    let (order, rest) = rest.split_once(", ")?;
    let (_returns_old, tail) = rest.split_once(", ")?;
    let plane = name.trim_matches('\'');
    let role = recorded_write_role(
        plane,
        if op == "'store'" {
            "wide_out"
        } else {
            DEFAULT_OLD_WRITE_ROLE
        },
    );
    if op == "'store'" {
        return Some(format!("('wide_write', {head}, ('{role}', {name}, {tail}"));
    }
    let contract = match (op, summation, order) {
        ("'or'", ..) => Some("or".to_string()),
        (_, _, "'lane'") => Some("ordered".to_string()),
        (_, "compensated", _) => Some(
            RECORDED_CONTRACTS
                .with(|state| state.borrow().get(plane).cloned())
                .unwrap_or_else(|| "compensated".to_string()),
        ),
        _ => None,
    };
    let stated = contract.map_or(String::new(), |c| format!("'{c}', "));
    Some(format!(
        "('accum_write', {head}, ('{role}', {name}, {stated}{tail}"
    ))
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
        assert_eq!(old_role_name("plane_in", "x", 0, false), "per_sample");
        assert_eq!(old_role_name("plane_in", "x", 1, false), "vec_in");
        assert_eq!(old_role_name("plane_in", "x", 2, false), "mat_in");
        assert_eq!(old_role_name("plane_in", "x", 3, false), "plane_in");
        assert_eq!(old_role_name("mutable", "x", 1, false), "mutable");
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

    /// Oracle: the rule written by hand: a declared slot that had the accumulate role says the runtime zeroes it, one that had
    /// the store role does not, and both take the one written-plane role.
    #[test]
    fn a_declared_slot_states_the_zeroing_its_old_role_meant() {
        let slot = |role: &str| {
            format!(r#"{{"r":"Slot","m":"m","v":{{"name":{{"s":"p"}},"role":{{"s":"{role}"}}}}}}"#)
        };
        let mut zeroed = parse(&slot("accum_out")).unwrap();
        assert!(migrate_roles(&mut zeroed) > 0);
        let text = render(&zeroed, "");
        assert!(
            text.contains("plane_out") && text.contains(r#""zeroed":true"#),
            "{text}"
        );
        let mut kept = parse(&slot("wide_out")).unwrap();
        migrate_roles(&mut kept);
        let text = render(&kept, "");
        assert!(
            text.contains("plane_out") && !text.contains("zeroed"),
            "{text}"
        );
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
        assert_eq!(old_role_name("plane_in", "x", 0, false), "vec_in");
        let two = parse(&format!(
            r#"{{"l":[{},{}]}}"#,
            leaf("vec_in"),
            leaf("per_sample")
        ))
        .unwrap();
        let why = role_conflict(&two).unwrap();
        assert!(
            why.contains("two planes named 'x'") && why.contains("'vec_in' and 'per_sample'"),
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

    /// Oracle: the key line the recording made of a sink of a kind that refines the store and states its own tail, a
    /// `wide_write` with a `wide_out` plane, as the stored digests spell it.
    #[test]
    fn a_kind_refining_the_store_reads_as_the_old_write_class_of_its_plane() {
        forget_recorded_roles();
        note_recorded_role("acc", "wide_out");
        assert_eq!(
            old_read_role("8|('update', 'int32', (), None, (2, 7), ('plane_out', 'acc', 'k'))"),
            "8|('wide_write', 'int32', (), None, (2, 7), ('wide_out', 'acc', 'k'))"
        );
    }
}
