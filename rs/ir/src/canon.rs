//! The canonical walk of a set of sinks: the traversal, the total merge of the bindings, the slot order, the access class, the
//! dispatch listing and the digest. Walks with loops, quantities, declared slots, dialect classes or raw blocks are refused.
use crate::door::{cstr, guard, put, slice};
use crate::record::{badarg, refuse, Ctx, Fault};
use crate::value::{parse, text_val, ttype_val, Val};
use rcore::{Rec, CONSTS, STR_CONSTS};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::ffi::{c_char, c_int};

/// One finished walk, owned by its context: the canonical node order and the text of each field.
pub struct WalkRes { pub order: Vec<u32>, pub fields: Vec<(&'static str, String)>, pub digest: String }

type Key = (String, String);

/// A TensorType's parts, as a binding carries them.
#[derive(Clone, PartialEq)]
struct Tt { dtype: String, shape: Vec<i64>, tag: Option<String> }
impl Tt {
    fn val(&self) -> Val { ttype_val(&self.dtype, &self.shape, self.tag.as_deref()) }
    fn text(&self) -> String { text_val(&self.val()) }
    /// From a TensorType record, or the shape, dtype and tag fields of a Wire.
    fn of(f: &[(String, Val)]) -> Result<Tt, Fault> {
        let get = |k: &str| f.iter().find(|x| x.0 == k).map(|x| &x.1);
        let shape = match get("shape") { Some(Val::Tuple(v)) => v.iter().map(|x| if let Val::Int(i) = x { Ok(*i as i64) } else { refuse("shape: expected ints") }).collect::<Result<_, _>>()?, None => vec![], _ => return refuse("shape: expected a tuple") };
        let Some(Val::Str(dtype)) = get("dtype") else { return refuse("dtype: expected a string") };
        Ok(Tt { dtype: dtype.clone(), shape, tag: if let Some(Val::Str(t)) = get("tag") { Some(t.clone()) } else { None } })
    }
}

/// A Quantity record: its namespace and its wires (name, role, type).
struct Quantity { prefix: String, wires: Vec<(String, String, Tt)> }
impl Quantity {
    fn of(v: &Val, sep: &str) -> Result<Option<Quantity>, Fault> {
        let Val::Rec(n, f) = v else { return Ok(None) };
        if n != "Quantity" { return refuse("a leaf's quantity is a Quantity record") }
        let get = |k: &str| f.iter().find(|x| x.0 == k).map(|x| &x.1);
        let Some(Val::Str(slug)) = get("slug") else { return refuse("Quantity.slug: expected a string") };
        let prefix = match get("name") { Some(Val::Str(nm)) => format!("{slug}{sep}{nm}"), _ => slug.clone() };
        let mut wires = vec![];
        if let Some(Val::Tuple(ws)) = get("wires") {
            for w in ws {
                let Val::Rec(_, wf) = w else { return refuse("Quantity.wires: expected Wire records") };
                let (Some((_, Val::Str(name))), Some((_, Val::Str(role)))) = (wf.iter().find(|x| x.0 == "name"), wf.iter().find(|x| x.0 == "role")) else { return refuse("Wire: name and role are strings") };
                wires.push((name.clone(), role.clone(), Tt::of(wf)?));
            }
        }
        Ok(Some(Quantity { prefix, wires }))
    }
    /// Every wire's slot, in declaration order.
    fn expand(&self, sep: &str) -> Vec<(Key, Tt)> { self.wires.iter().map(|w| ((w.1.clone(), format!("{}{sep}{}", self.prefix, w.0)), w.2.clone())).collect() }
}

/// The declared slots: a tuple of Slot records.
fn declared_slots(text: Option<&str>) -> Result<Vec<(String, String, Tt)>, Fault> {
    let (Val::Tuple(v) | Val::List(v)) = parse(text.unwrap_or("()")).map_err(Fault::Refused)? else { return refuse("declared: expected a tuple of Slot records") };
    v.iter().map(|x| {
        let Val::Rec(_, f) = x else { return refuse("declared: expected Slot records") };
        let get = |k: &str| f.iter().find(|x| x.0 == k).map(|x| &x.1);
        let (Some(Val::Str(r)), Some(Val::Str(n)), Some(Val::Rec(_, t))) = (get("role"), get("name"), get("ttype")) else { return refuse("Slot: role, name and ttype") };
        Ok((r.clone(), n.clone(), Tt::of(t)?))
    }).collect()
}

fn s(t: &str) -> Val { Val::Str(t.into()) }
fn pair(a: &str, b: &str) -> Val { Val::Tuple(vec![s(a), s(b)]) }
fn str_const(k: &str) -> &'static str { STR_CONSTS.iter().find(|c| c.0 == k).map_or("", |c| c.1) }
fn not_built<T>(what: &str) -> Result<T, Fault> { refuse(format!("not in this build: {what}")) }
fn rep(t: &str) -> String { rcore::keytext::py_str(t) }

fn is_empty_seq(text: Option<&str>) -> Result<bool, Fault> {
    match parse(text.unwrap_or("()")).map_err(Fault::Refused)? {
        Val::Tuple(v) | Val::List(v) => Ok(v.is_empty()),
        _ => refuse("expected a tuple"),
    }
}

impl Ctx {
    pub(crate) fn is_a(&self, h: u32, class: &str) -> bool { self.table.is_a(self.nodes[h as usize].class, class) }
    fn tt(&self, h: u32) -> Tt { let n = &self.nodes[h as usize]; Tt { dtype: n.dtype.clone(), shape: n.shape.clone(), tag: n.tag.clone() } }

    pub fn canonical(&mut self, sinks: &[u32], declared: Option<&str>, force: Option<&str>, exempt: Option<&str>) -> Result<usize, Fault> {
        if sinks.is_empty() { return refuse("canonical(): the sink set is empty: the sinks are the walk's root set") }
        for &h in sinks {
            self.node(h)?;
            if self.is_a(h, "Loop") {
                if self.refs(h, "body_sinks")?.is_empty() {
                    let ix = self.node_idx_name(h)?;
                    return refuse(format!("canonical(): the loop over {} is in the root set but commits nothing; a loop is a root only when its body commits a sink once per iteration", rep(&ix)))
                }
                continue;
            }
            if !self.is_a(h, "Sink") { return refuse(format!("canonical(): a {} is not a sink; the root set is sinks only", self.table.classes[self.nodes[h as usize].class].name)) }
        }
        let exempt: Vec<String> = match parse(exempt.unwrap_or("()")).map_err(Fault::Refused)? {
            Val::Tuple(v) | Val::List(v) => v.into_iter().filter_map(|x| if let Val::Str(s) = x { Some(s) } else { None }).collect(),
            _ => return refuse("exempt planes: expected a tuple of str"),
        };
        let finishes: Vec<u32> = sinks.iter().copied().filter(|&h| self.is_a(h, "Finish")).collect();
        if finishes.len() > 1 { return refuse(format!("canonical(): {} finish statements; a kernel finishes its sample through ONE mask", finishes.len())) }
        let finish = match finishes.first() { Some(&f) => self.field(f, "name")?, None => Val::None };

        // the traversal: records numbered by discovery, one per node object
        let mut ids: HashMap<u32, u32> = HashMap::new();
        let (mut objs, mut recs): (Vec<u32>, Vec<Rec>) = (vec![], vec![]);
        let mut stack: Vec<u32> = sinks.to_vec();
        while let Some(h) = stack.pop() {
            if ids.contains_key(&h) { continue }
            let n = self.node(h)?;
            let c = &self.table.classes[n.class];
            ids.insert(h, objs.len() as u32);
            let r = self.rec_of(h)?;
            stack.extend(n.ops.iter().copied());
            objs.push(h); recs.push(r);
        }
        for (i, &h) in objs.iter().enumerate() { recs[i].ops = self.nodes[h as usize].ops.iter().map(|o| ids[o]).collect() }
        let roots: Vec<u32> = sinks.iter().map(|h| ids[h]).collect();
        let w = rcore::walk_full(&recs, &roots).map_err(Fault::Refused)?;
        // the flags of each record: binds a slot, is watched by the access and dispatch passes
        let class_of = |i: u32| &self.table.classes[self.nodes[objs[i as usize] as usize].class];
        let watched_of = |i: u32| { let c = class_of(i); c.watch && (c.decl.is_some() || c.spec.is_some_and(|s| s.watch == 1) || recs[i as usize].kind == "raw_device") };

        // the total merge of the bindings, in visit order; a canonical node is the first of its key
        let mut held: Vec<(Key, Tt, String)> = vec![];
        let mut at: HashMap<Key, usize> = HashMap::new();
        let mut pairs: Vec<(u32, u32)> = vec![];
        for &v in &w.visits {
            let (node, canon) = (objs[v as usize], objs[w.order[w.pos[v as usize] as usize] as usize]);
            if class_of(v).binds { pairs.push((node, canon)) }
            if self.is_a(node, "Loop") { for sk in self.refs(node, "body_sinks")? { pairs.push((sk, sk)) } }
        }
        for (node, canon) in pairs {
            let key = (self.str_field(node, "role")?, self.str_field(node, "name")?);
            match at.get(&key) {
                None => { at.insert(key.clone(), held.len()); held.push((key, self.tt(canon), self.table.classes[self.nodes[canon as usize].class].name.clone())) }
                Some(&p) => {
                    let (a, b) = (&held[p].1, self.tt(node));
                    if *a != b {
                        return refuse(format!("leaf identity collision on (role={}, name={}): {} is typed {} but {} is typed {}. A slot is keyed on (role, name) ALONE, so these two \
would share one slot; the merge is TOTAL and the disagreement is refused, never silently collapsed.", rep(&key.0), rep(&key.1),
                            held[p].2, a.text(), self.table.classes[self.nodes[node as usize].class].name, b.text()))
                    }
                }
            }
        }
        // declared slots: a read is minted when the body binds none, a sink slot is kept for the order
        let sink_roles = CONSTS.iter().find(|c| c.0 == "SINK_ROLES").map_or(&[][..], |c| c.1);
        let mut declared_sinks: Vec<(Key, Tt)> = vec![];
        for (role, name, tt) in declared_slots(declared)? {
            let key = (role.clone(), name.clone());
            match at.get(&key) {
                None if sink_roles.contains(&role.as_str()) => declared_sinks.push((key, tt)),
                None => { at.insert(key.clone(), held.len()); held.push((key, tt, "Leaf".into())) }
                Some(&p) => if held[p].1 != tt {
                    return refuse(format!("declared slot (role={}, name={}) is typed {} but the body binds it as {}; a slot is keyed on (role, name) ALONE", rep(&role), rep(&name), tt.text(), held[p].1.text()))
                }
            }
        }
        // the quantities reached, by namespace, with the all-or-nothing expansion
        let sep = str_const("NAMESPACE_SEP");
        let mut reached: Vec<(String, Quantity)> = vec![];
        for &i in &w.order {
            let h = objs[i as usize];
            if !self.is_a(h, "Leaf") { continue }
            let Some(q) = Quantity::of(&self.field(h, "quantity")?, sep)? else { continue };
            match reached.iter().find(|r| r.0 == q.prefix) {
                None => reached.push((q.prefix.clone(), q)),
                Some((_, held)) => if held.wires != q.wires {
                    let names = |q: &Quantity| q.wires.iter().map(|x| rep(&x.1)).collect::<Vec<_>>().join(", ");
                    return refuse(format!("two quantities share the namespace {} but declare different wires ([{}] vs [{}]) - a namespace identifies a quantity ", rep(&q.prefix), names(held), names(&q)))
                }
            }
        }
        let mut owner: HashMap<Key, (usize, usize)> = HashMap::new();
        let mut slots: Vec<(Key, Tt)> = vec![];
        let mut pos_of: HashMap<Key, usize> = HashMap::new();
        for (qi, (_, q)) in reached.iter().enumerate() {
            for (i, s) in q.expand(sep).into_iter().enumerate() {
                owner.insert(s.0.clone(), (qi, i));
                match pos_of.get(&s.0) { Some(&p) => slots[p] = s, None => { pos_of.insert(s.0.clone(), slots.len()); slots.push(s) } }
            }
        }
        for (key, tt) in declared_sinks { if !pos_of.contains_key(&key) { pos_of.insert(key.clone(), slots.len()); slots.push((key, tt)) } }
        for (key, tt, _) in &held { if !pos_of.contains_key(key) { pos_of.insert(key.clone(), slots.len()); slots.push((key.clone(), tt.clone())) } }
        if let Some(((role, name), _)) = slots.iter().find(|(k, _)| k.1.contains(sep) && !owner.contains_key(k)) {
            return refuse(format!("wire {} (role {}) is bound with no owning quantity: the {} namespace is reserved for compound-quantity wires, which only a Quantity declaration may mint ", rep(name), rep(role), rep(sep)))
        }
        // the role order: quantity wires in declaration order, then the rest by name
        let roles = CONSTS.iter().find(|c| c.0 == "ROLE_ORDER").map_or(&[][..], |c| c.1);
        let mut ordered: Vec<&(Key, Tt)> = vec![];
        for role in roles {
            let (mut owned, mut free): (Vec<_>, Vec<_>) = slots.iter().filter(|(k, _)| k.0 == *role).partition(|(k, _)| owner.contains_key(k));
            owned.sort_by_key(|(k, _)| owner[k]);
            free.sort_by(|a, b| (a.0).1.cmp(&(b.0).1));
            ordered.extend(owned); ordered.extend(free);
        }
        if let Some(((r, _), _)) = slots.iter().find(|(k, _)| !roles.contains(&k.0.as_str())) { return refuse(format!("binding with a role outside the canonical order: {r}")) }
        let slot_of: HashMap<&Key, usize> = ordered.iter().enumerate().map(|(i, (k, _))| (k, i)).collect();
        // the spans: one consecutive, declaration-ordered run per role
        let mut spans: Vec<(String, String)> = vec![];
        for (prefix, q) in &reached {
            let expanded = q.expand(sep);
            let (mut idx, mut per): (Vec<i128>, Vec<(String, Vec<i128>)>) = (vec![], vec![]);
            for (k, _) in &expanded {
                let i = slot_of[k] as i128;
                idx.push(i);
                match per.iter_mut().find(|p| p.0 == k.0) { Some(p) => p.1.push(i), None => per.push((k.0.clone(), vec![i])) }
            }
            for (role, run) in &per {
                let mut sorted = run.clone(); sorted.sort();
                if *run != sorted || run.iter().enumerate().any(|(j, v)| *v != run[0] + j as i128) {
                    return refuse(format!("quantity {} occupies a non-contiguous or mis-ordered span in role {}: arg_spec indices {:?} (require one consecutive, declaration-ordered run per role)", rep(prefix), rep(role), run))
                }
            }
            let span = Val::Rec("QuantitySpan".into(), vec![("prefix".into(), s(prefix)), ("wires".into(), Val::Tuple(expanded.iter().map(|(k, _)| s(&k.1)).collect())),
                ("slots".into(), Val::Tuple(idx.into_iter().map(Val::Int).collect()))]);
            spans.push((prefix.clone(), text_val(&span)));
        }
        spans.sort();
        // the access class over the watched nodes
        let rank_of = |a: &str| rcore::facts::ACCESS_RANK.iter().position(|x| *x == a).unwrap_or(0);
        let watched: Vec<u32> = w.order.iter().copied().filter(|&i| watched_of(i)).map(|i| objs[i as usize]).collect();
        let mut all = vec![];
        self.with_loop_bodies(&watched, &mut all)?;
        let mut seam = HashSet::new();
        for &h in &all { if self.is_a(h, "Loop") { if let Some(x) = self.step_word(self.node_field_pub(h, "count")?, &exempt)? { seam.insert(x); } } }
        let mut cls = 0usize;
        for &h in &all {
            let param_cell = self.is_a(h, "At") && {
                let i = self.nodes[h as usize].ops[1];
                self.is_a(i, "RoleConst") && self.field(i, "label")? == s(str_const("PARAM_CELL"))
            };
            if seam.contains(&h) || param_cell { continue }
            let new = if self.is_a(h, "Op") && self.str_field(h, "kind")? == "raw_device" {
                let lit = self.field(h, "literal")?;
                let acc = if let Val::Rec(_, f) = &lit { f.iter().find(|x| x.0 == "access").map(|x| x.1.clone()) } else { None };
                match &acc { Some(Val::Str(a)) if rcore::facts::ACCESS_RANK.contains(&a.as_str()) => rank_of(a),
                    _ => return refuse(format!("a spliced raw block reached the classifier declaring access={}: @raw_device requires an explicit access= annotation \\\ndrawn from {:?} — the walk cannot see inside the text, and a mapreduce class is a sink property, not a block's",
                        acc.map_or("None".into(), |a| text_val(&a)), rcore::facts::ACCESS_RANK)) }
            } else if self.is_a(h, "At") { 1 } else if (self.is_a(h, "WideWrite") || self.is_a(h, "AccumWrite")) && self.nodes[h as usize].ops.len() > 1 { 2 } else { 0 };
            cls = cls.max(new);
        }
        let mut access = rcore::facts::ACCESS_RANK[cls];
        if let Some(f) = force { if rank_of(f) > cls { access = rcore::facts::ACCESS_RANK[rank_of(f)] } }
        // the dispatch listing: selector reads, and whether the selector is static
        let mut written: HashSet<String> = HashSet::new();
        for &h in &watched {
            if self.is_a(h, "Sink") { written.insert(self.str_field(h, "name")?); }
            if self.is_a(h, "Loop") { for sk in self.refs(h, "body_sinks")? { if self.is_a(sk, "Sink") { written.insert(self.str_field(sk, "name")?); } } }
        }
        let mut listing = vec![];
        for (i, &h) in self.dispatch_nodes(&watched)?.iter().enumerate() {
            let (mut seen, mut reads, mut st) = (HashSet::new(), BTreeSet::new(), vec![self.nodes[h as usize].ops[0]]);
            while let Some(n) = st.pop() {
                if !seen.insert(n) { continue }
                if self.is_a(n, "Leaf") { reads.insert((self.str_field(n, "role")?, self.str_field(n, "name")?)); }
                else if self.is_a(n, "LoopCarry") { reads.insert(("mutable".to_string(), self.str_field(n, "name")?)); }
                st.extend(self.nodes[n as usize].ops.iter().copied());
            }
            let stat = !reads.iter().any(|(_, nm)| written.contains(nm));
            listing.push(Val::Tuple(vec![s(&format!("dispatch{i}")), self.field(h, "K")?, Val::Tuple(reads.iter().map(|(a, b)| pair(a, b)).collect()), Val::Bool(stat)]));
        }
        let mut prior = BTreeSet::new();
        for &i in &w.order {
            let h = objs[i as usize];
            if self.is_a(h, "Leaf") && self.str_field(h, "kind")? == "prior_read" { prior.insert(self.str_field(h, "name")?); }
        }
        // the digest tail and the fields
        let (mut tail, mut leaves, mut spec, mut types, mut index) = (String::new(), vec![], vec![], vec![], vec![]);
        for (i, ((role, name), tt)) in ordered.iter().map(|x| (&x.0, &x.1)).enumerate() {
            tail.push_str(&format!("leaf|{role}|{name}|{}\n", tt.text()));
            leaves.push(Val::Rec("Slot".into(), vec![("role".into(), s(role)), ("name".into(), s(name)), ("ttype".into(), tt.val())]));
            spec.push(pair(role, name));
            types.push(Val::Tuple(vec![pair(role, name), tt.val()]));
            index.push(Val::Tuple(vec![pair(role, name), Val::Int(i as i128)]));
        }
        tail.push_str(&format!("arg_spec|{}\naccess|{access}|None\n", text_val(&Val::Tuple(spec.clone()))));
        for (_, q) in &spans { tail.push_str(&format!("quantity|{q}\n")) }
        let digest = rcore::digest(&w.pre, tail.as_bytes());
        let t = |v: Vec<Val>| text_val(&Val::Tuple(v));
        let fields = vec![("leaves", t(leaves)), ("arg_spec", t(spec)), ("slot_of", t(index)), ("slot_types", t(types)), ("access", text_val(&s(access))),
            ("quantities", format!("({}{})", spans.iter().map(|(p, q)| format!("({}, {q})", rep(p))).collect::<Vec<_>>().join(", "), if spans.len() == 1 { "," } else { "" })), ("digest", text_val(&s(&digest))), ("dispatches", t(listing)),
            ("prior", t(prior.iter().map(|p| s(p)).collect())), ("finish", text_val(&finish)), ("preimage", w.pre.clone())];
        self.walks.push(WalkRes { order: w.order.iter().map(|&i| objs[i as usize]).collect(), fields, digest });
        Ok(self.walks.len() - 1)
    }

    fn node_idx_name(&self, lp: u32) -> Result<String, Fault> {
        match self.field(lp, "index")? { Val::Node(i) => self.str_field(i, "name"), _ => refuse("a loop's index is a node") }
    }
    fn node_field_pub(&self, h: u32, f: &str) -> Result<Option<u32>, Fault> {
        match self.field(h, f)? { Val::Node(n) => Ok(Some(n)), _ => Ok(None) }
    }

    fn walk(&self, w: u32) -> Result<&WalkRes, Fault> { self.walks.get(w as usize).map_or_else(|| refuse(format!("walk handle {w} is not in this context")), Ok) }
}

fn opt<'a>(p: *const c_char, what: &str) -> Result<Option<&'a str>, Fault> { if p.is_null() { Ok(None) } else { cstr(p, what).map(Some) } }

#[no_mangle]
pub extern "C" fn rx_canonical(ctx: *mut Ctx, sinks: *const u32, n: usize, declared_kt: *const c_char, force_access: *const c_char,
                               exempt_kt: *const c_char, out: *mut u32) -> c_int {
    guard(ctx, |c| {
        if out.is_null() { return badarg("out is NULL") }
        let w = c.canonical(slice(sinks, n, "sinks")?, opt(declared_kt, "declared_kt")?, opt(force_access, "force_access")?, opt(exempt_kt, "exempt_kt")?)?;
        unsafe { *out = w as u32 }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn rx_walk_order(ctx: *mut Ctx, w: u32, buf: *mut u32, cap: usize, n: *mut usize) -> c_int {
    guard(ctx, |c| {
        if n.is_null() { return badarg("n is NULL") }
        let o = &c.walk(w)?.order;
        unsafe { *n = o.len() }
        if cap < o.len() || (buf.is_null() && !o.is_empty()) { return Err(Fault::Truncated) }
        unsafe { std::ptr::copy_nonoverlapping(o.as_ptr(), buf, o.len()) }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn rx_walk_field(ctx: *mut Ctx, w: u32, field: *const c_char, buf: *mut c_char, cap: usize, len: *mut usize) -> c_int {
    guard(ctx, |c| {
        let f = cstr(field, "field")?;
        let t = c.walk(w)?.fields.iter().find(|x| x.0 == f).map(|x| x.1.clone());
        put(buf, cap, len, &t.map_or_else(|| refuse(format!("a walk has no field {f:?}")), Ok)?)
    })
}

#[no_mangle]
pub extern "C" fn rx_digest(ctx: *mut Ctx, w: u32, out65: *mut c_char) -> c_int {
    guard(ctx, |c| {
        if out65.is_null() { return badarg("out65 is NULL") }
        let d = &c.walk(w)?.digest;
        unsafe { std::ptr::copy_nonoverlapping(d.as_ptr(), out65 as *mut u8, d.len()); *out65.add(d.len()) = 0 }
        Ok(())
    })
}
