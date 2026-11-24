//! The canonical walk of a set of sinks: the traversal, the total merge of the bindings, the slot order, the access class, the
//! dispatch listing and the digest. Walks with loops, quantities, declared slots, dialect classes are refused.
use crate::door::{cstr, guard, put, slice};
use crate::record::{badarg, refuse, Ctx, Fault};
use crate::value::{parse, text_val, ttype_val, Val};
use rcore::{Rec, CONSTS, STR_CONSTS};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ffi::{c_char, c_int};

/// One finished walk, owned by its context: the canonical node order and the text of each field.
pub struct WalkRes {
    /// The canonical position of each node the traversal reached.
    pub positions: HashMap<u32, u32>,
    /// The nodes in canonical order.
    pub order: Vec<u32>,
    /// The slots the walk binds, each with its key and type, in argument order.
    pub leaves: Vec<(Key, Tt)>,
    /// The written-plane slots the runtime zeroes before the run: a sink on the plane accumulates, sums samples or is an
    /// ordered accumulate; a plane only plain stores write is left as the caller gave it.
    pub zeroed: HashSet<Key>,
    /// The counting planes the walk's slots state, each as `(count plane, the appended plane it counts)`, in the caller's
    /// order: the runtime makes the count with the run and hands it back beside its plane.
    pub counts: Vec<(String, String)>,
    /// The winner planes the walk's slots state, each as `(winner plane, the plane it ranks)`, in the caller's order: the
    /// runtime makes the winner with the run and hands it back beside its plane.
    pub winners: Vec<(String, String)>,
    /// The access class of the walk.
    pub access: String,
    /// The name of the finish statement among the sinks, when there is one.
    pub finish: Option<String>,
    /// The names of the planes whose launch-start value the walk reads, sorted.
    pub prior: Vec<String>,
    /// For each dispatch of the walk, whether no sink of the walk writes a plane its selector reads.
    pub static_selector: Vec<bool>,
    /// For each dispatch of the walk, the `(role, name)` slots its selector reads, sorted.
    pub selector_reads: Vec<Vec<Key>>,
    /// The quantities the walk reaches, in walk order: namespace, the slot name of each wire and the slot number of each wire.
    pub quantities: Vec<QuantityRow>,
    /// The text of each field of the walk, by name.
    pub fields: Vec<(&'static str, String)>,
    /// The digest of the walk.
    pub digest: String,
}

/// One reached quantity of a walk: its namespace, the slot names of its wires and their slot numbers, in declaration order.
pub type QuantityRow = (String, Vec<String>, Vec<usize>);

/// The spans of the reached quantities as text, and the reached quantities as rows.
type Spans = (Vec<(String, String)>, Vec<QuantityRow>);

/// What the dispatch listing holds: the listing values, the static flag and the selector reads of each dispatch.
type Listing = (Vec<Val>, Vec<bool>, Vec<Vec<Key>>);

/// A slot key: role and name.
pub type Key = (String, String);

/// A TensorType's parts, as a binding carries them.
#[derive(Clone, PartialEq)]
pub struct Tt {
    /// The element type name.
    pub dtype: String,
    /// The extents.
    pub shape: Vec<i64>,
    /// The tag, if any.
    pub tag: Option<String>,
    /// The axis list a plane leaf declares, as text; empty for a type bound without axes.
    pub axes: String,
}
impl Tt {
    /// The text of the type and of the axes a plane declares, as the digest states a slot.
    fn slot_text(&self) -> String {
        if self.axes.is_empty() {
            self.text()
        } else {
            format!("{}|axes={}", self.text(), self.axes)
        }
    }

    /// The type as a TensorType value.
    fn val(&self) -> Val {
        ttype_val(&self.dtype, &self.shape, self.tag.as_deref())
    }
    /// The type as text.
    fn text(&self) -> String {
        text_val(&self.val())
    }
    /// From a TensorType record, or the shape, dtype and tag fields of a Wire.
    fn of(f: &[(String, Val)]) -> Result<Tt, Fault> {
        let get = |k: &str| f.iter().find(|x| x.0 == k).map(|x| &x.1);
        let shape = match get("shape") {
            Some(Val::Tuple(v)) => v
                .iter()
                .map(|x| {
                    if let Val::Int(i) = x {
                        Ok(*i as i64)
                    } else {
                        refuse("shape: expected ints")
                    }
                })
                .collect::<Result<_, _>>()?,
            None => vec![],
            _ => return refuse("shape: expected a tuple"),
        };
        let Some(Val::Str(dtype)) = get("dtype") else {
            return refuse("dtype: expected a string");
        };
        Ok(Tt {
            dtype: dtype.clone(),
            shape,
            tag: if let Some(Val::Str(t)) = get("tag") {
                Some(t.clone())
            } else {
                None
            },
            axes: String::new(),
        })
    }
}

/// A Quantity record: its namespace and its wires (name, role, type).
struct Quantity {
    /// The namespace of the quantity.
    prefix: String,
    /// Each wire: name, role and type.
    wires: Vec<(String, String, Tt)>,
}
impl Quantity {
    /// Reads a Quantity record; None when the value is not one.
    fn of(v: &Val, sep: &str) -> Result<Option<Quantity>, Fault> {
        let Val::Rec(n, f) = v else { return Ok(None) };
        if n != "Quantity" {
            return refuse("a leaf's quantity is a Quantity record");
        }
        let get = |k: &str| f.iter().find(|x| x.0 == k).map(|x| &x.1);
        let Some(Val::Str(slug)) = get("slug") else {
            return refuse("Quantity.slug: expected a string");
        };
        let prefix = match get("name") {
            Some(Val::Str(nm)) => format!("{slug}{sep}{nm}"),
            _ => slug.clone(),
        };
        let mut wires = vec![];
        if let Some(Val::Tuple(ws)) = get("wires") {
            for w in ws {
                let Val::Rec(_, wf) = w else {
                    return refuse("Quantity.wires: expected Wire records");
                };
                let (Some((_, Val::Str(name))), Some((_, Val::Str(role)))) = (
                    wf.iter().find(|x| x.0 == "name"),
                    wf.iter().find(|x| x.0 == "role"),
                ) else {
                    return refuse("Wire: name and role are strings");
                };
                wires.push((name.clone(), role.clone(), Tt::of(wf)?));
            }
        }
        Ok(Some(Quantity { prefix, wires }))
    }
    /// Every wire's slot, in declaration order.
    fn expand(&self, sep: &str) -> Vec<(Key, Tt)> {
        self.wires
            .iter()
            .map(|w| {
                (
                    (w.1.clone(), format!("{}{sep}{}", self.prefix, w.0)),
                    w.2.clone(),
                )
            })
            .collect()
    }
}

/// The slots a caller declares for a walk: each with its type, and which of them are written planes the runtime zeroes.
#[derive(Clone, Default)]
pub struct Declared {
    /// The declared slots, in the caller's order.
    pub slots: Vec<(Key, Tt)>,
    /// The written planes among them that the runtime zeroes before the run.
    pub zeroed: HashSet<Key>,
    /// The counting planes a record states (`counts='<plane>'`), each as `(count plane, the plane it counts)`.
    pub counts: Vec<(String, String)>,
    /// The winner planes a record states (`winner_of='<plane>'`), each as `(winner plane, the plane it ranks)`.
    pub winners: Vec<(String, String)>,
}

/// The declared slots: a tuple of Slot records. A record may state `zeroed=True` for a written plane no sink of the walk writes
/// by accumulating (a companion plane the runtime folds), `counts='<plane>'` for a one-element plane that counts the
/// appends into `<plane>`, and `winner_of='<plane>'` for a plane that records, per element of `<plane>`, the lane whose value a
/// minimum or maximum kept.
pub fn declared_slots(text: Option<&str>) -> Result<Declared, Fault> {
    let (Val::Tuple(v) | Val::List(v)) = parse(text.unwrap_or("()")).map_err(Fault::Refused)?
    else {
        return refuse("declared: expected a tuple of Slot records");
    };
    let mut declared = Declared::default();
    for x in v.iter() {
        {
            let Val::Rec(_, f) = x else {
                return refuse("declared: expected Slot records");
            };
            let get = |k: &str| f.iter().find(|x| x.0 == k).map(|x| &x.1);
            let (Some(Val::Str(r)), Some(Val::Str(n)), Some(Val::Rec(_, t))) =
                (get("role"), get("name"), get("ttype"))
            else {
                return refuse("Slot: role, name and ttype");
            };
            let key = (r.clone(), n.clone());
            if matches!(get("zeroed"), Some(Val::Bool(true))) {
                declared.zeroed.insert(key.clone());
            }
            if let Some(Val::Str(of)) = get("counts") {
                declared.counts.push((key.1.clone(), of.clone()));
            }
            if let Some(Val::Str(of)) = get("winner_of") {
                declared.winners.push((key.1.clone(), of.clone()));
            }
            declared.slots.push((key, Tt::of(t)?));
        }
    }
    Ok(declared)
}

/// A text value.
fn s(t: &str) -> Val {
    Val::Str(t.into())
}
/// A pair of texts as a tuple value.
fn pair(a: &str, b: &str) -> Val {
    Val::Tuple(vec![s(a), s(b)])
}
/// Refuses a count that names a plane the walk does not bind, or whose own plane is not a written one.
fn check_counts(counts: &[(String, String)], slots: &[(Key, Tt)]) -> Result<(), Fault> {
    let bound = |name: &str| slots.iter().find(|(key, _)| key.1 == name);
    for (count, of) in counts {
        let (Some((counted, _)), Some(_)) = (bound(count), bound(of)) else {
            return refuse(format!(
                "counts: {count:?} counts {of:?}, and the walk binds both planes or neither"
            ));
        };
        if counted.0 != "plane_out" {
            return refuse(format!(
                "counts: {count:?} is a count, which the runtime writes: a written plane, not {:?}",
                counted.0
            ));
        }
    }
    Ok(())
}
/// Refuses a winner plane that ranks a plane the walk does not bind, that is not a written unsigned plane, or that ranks a plane of
/// vectors (a winner is one lane per element).
fn check_winners(winners: &[(String, String)], slots: &[(Key, Tt)]) -> Result<(), Fault> {
    let bound = |name: &str| slots.iter().find(|(key, _)| key.1 == name);
    for (winner, of) in winners {
        let (Some((key, tt)), Some((_, ranked))) = (bound(winner), bound(of)) else {
            return refuse(format!(
                "winner_of: {winner:?} ranks {of:?}, and the walk binds both planes or neither"
            ));
        };
        if key.0 != "plane_out" || tt.dtype != "uint32" || !tt.shape.is_empty() {
            return refuse(format!(
                "winner_of: {winner:?} records lanes, which the runtime writes: a written scalar uint32 plane, not {:?} {}",
                key.0,
                tt.text()
            ));
        }
        if !ranked.shape.is_empty() {
            return refuse(format!(
                "winner_of: {winner:?} ranks one lane per element, and {of:?} holds {}",
                ranked.text()
            ));
        }
    }
    Ok(())
}
/// The text constant `k` of the core, or empty.
pub(crate) fn str_const(k: &str) -> &'static str {
    STR_CONSTS.iter().find(|c| c.0 == k).map_or("", |c| c.1)
}

impl Ctx {
    /// The type of node `h`.
    fn tt(&self, h: u32) -> Tt {
        let n = &self.nodes[h as usize];
        let axes = match self.field(h, "axes") {
            Ok(Val::Tuple(declared)) if !declared.is_empty() => text_val(&Val::Tuple(declared)),
            _ => String::new(),
        };
        Tt {
            dtype: n.dtype.clone(),
            shape: n.shape.clone(),
            tag: n.tag.clone(),
            axes,
        }
    }

    /// The canonical walk of the sinks (and of the loop bodies they commit), stored in the context: the sink checks, the
    /// traversal, the merge of the bindings, the slot order and spans, the access class, the dispatch listing and the digest.
    /// Returns the walk's handle; refuses an empty or non-sink root set and any binding conflict.
    pub fn canonical(
        &mut self,
        sinks: &[u32],
        declared: &Declared,
        force: Option<&str>,
        exempt: &[String],
    ) -> Result<usize, Fault> {
        self.check_roots(sinks)?;
        let finish = self.finish_name(sinks)?;
        let tr = self.traverse(sinks)?;
        let mut bound = self.merge_bindings(&tr)?;
        let declared_sinks = merge_declared(&mut bound, declared)?;
        let sep = str_const("NAMESPACE_SEP");
        let reached = self.reached_quantities(&tr, sep)?;
        let layout = place_slots(&reached, declared_sinks, &bound, sep)?;
        let ordered = order_slots(&layout)?;
        check_counts(&declared.counts, &ordered)?;
        check_winners(&declared.winners, &ordered)?;
        let (spans, quantities) = quantity_spans(&reached, &ordered, &layout.zeroed, sep)?;
        let watched = self.watched_nodes(&tr);
        let access = self.access_class(&watched, exempt, force)?;
        let (listing, static_selector, selector_reads) = self.dispatch_listing(&watched)?;
        let prior = self.prior_reads(&tr)?;
        #[cfg(feature = "replay-key-form")]
        let pre = tr.walk.pre.lines().fold(String::new(), |text, line| {
            text + &self.keyed_line(line) + "\n"
        });
        #[cfg(not(feature = "replay-key-form"))]
        let pre = tr.walk.pre.as_str();
        let taken_roles: Vec<String> = ordered
            .iter()
            .map(|(key, tt)| {
                let zeroed = layout.zeroed.contains(key);
                self.digest_role(&key.0, &key.1, tt.shape.len(), zeroed)
            })
            .collect();
        let keys = DigestKeys {
            pre: &pre,
            roles: &taken_roles,
        };
        let (fields, digest) =
            render_fields(&ordered, &spans, access, keys, listing, &prior, &finish);
        let (objs, w) = (&tr.objs, &tr.walk);
        self.walks.push(WalkRes {
            leaves: ordered,
            zeroed: layout.zeroed,
            counts: declared.counts.clone(),
            winners: declared.winners.clone(),
            access: access.to_string(),
            finish: match finish {
                Val::Str(name) => Some(name),
                _ => None,
            },
            prior: prior.into_iter().collect(),
            static_selector,
            selector_reads,
            quantities,
            positions: objs
                .iter()
                .enumerate()
                .map(|(i, &h)| (h, w.pos[i]))
                .collect(),
            order: w.order.iter().map(|&i| objs[i as usize]).collect(),
            fields,
            digest,
        });
        Ok(self.walks.len() - 1)
    }

    /// Checks that every root is a sink, or a loop that commits something; refuses otherwise.
    fn check_roots(&self, sinks: &[u32]) -> Result<(), Fault> {
        if sinks.is_empty() {
            return refuse("canonical(): the sink set is empty: the sinks are the walk's root set");
        }
        for &h in sinks {
            self.node(h)?;
            if self.is_a(h, "Loop") {
                if self.refs(h, "body_sinks")?.is_empty() {
                    let ix = self.node_idx_name(h)?;
                    return refuse(format!("canonical(): the loop over {} is in the root set but commits nothing; a loop is a root only when its body commits a sink once per iteration", rcore::keytext::py_str(&ix)));
                }
                continue;
            }
            if self.is_a(h, "Group") {
                continue;
            }
            if let Some(fault) = self.outside_group_fault(h) {
                return refuse(format!("canonical(): {fault}"));
            }
            if !self.is_a(h, "Sink") {
                return refuse(format!(
                    "canonical(): a {} is not a sink; the root set is sinks only",
                    self.table.classes[self.nodes[h as usize].class].name
                ));
            }
        }
        if sinks.iter().filter(|&&h| self.is_a(h, "Group")).count() > 1 {
            return refuse(
                "canonical(): a kernel holds one Group; two groups in one walk are not written",
            );
        }
        Ok(())
    }

    /// The name of the one finish statement among the sinks, or None; refuses more than one.
    fn finish_name(&self, sinks: &[u32]) -> Result<Val, Fault> {
        let finishes: Vec<u32> = sinks
            .iter()
            .copied()
            .filter(|&h| self.is_a(h, "Finish"))
            .collect();
        if finishes.len() > 1 {
            return refuse(format!(
                "canonical(): {} finish statements; a kernel finishes its sample through ONE mask",
                finishes.len()
            ));
        }
        let finish = match finishes.first() {
            Some(&f) => self.field(f, "name")?,
            None => Val::None,
        };
        Ok(finish)
    }

    /// The traversal: records numbered by discovery, one per node object, and the walk of those records.
    fn traverse(&self, sinks: &[u32]) -> Result<Traversal, Fault> {
        let mut ids: HashMap<u32, u32> = HashMap::new();
        let (mut objs, mut recs): (Vec<u32>, Vec<Rec>) = (vec![], vec![]);
        let mut stack: Vec<u32> = sinks.to_vec();
        while let Some(h) = stack.pop() {
            if ids.contains_key(&h) {
                continue;
            }
            self.node(h)?;
            ids.insert(h, objs.len() as u32);
            let mut r = self.rec_of(h)?;
            let reads = self.walked_operands(h)?;
            stack.extend(reads.iter().copied());
            if self.is_a(h, "Group") {
                r.rule = None; // its body sinks are walked as operands beyond the key the rule counts
            }
            objs.push(h);
            recs.push(r);
        }
        for (i, &h) in objs.iter().enumerate() {
            recs[i].ops = self.walked_operands(h)?.iter().map(|o| ids[o]).collect()
        }
        let roots: Vec<u32> = sinks.iter().map(|h| ids[h]).collect();
        let w = rcore::walk_full(&recs, &roots).map_err(Fault::Refused)?;
        Ok(Traversal { objs, walk: w })
    }

    /// The nodes the walk reads from `h`: its operands, and for a Group its body sinks after the key, so that the planes the body
    /// reads and writes are bound like any others and the body is part of the identity of the group.
    fn walked_operands(&self, h: u32) -> Result<Vec<u32>, Fault> {
        let mut reads = self.nodes[h as usize].ops.clone();
        if self.is_a(h, "Group") {
            reads.extend(self.refs(h, "body_sinks")?);
        }
        Ok(reads)
    }

    /// The class of the node a traversal index stands for.
    fn class_of_visit(&self, tr: &Traversal, i: u32) -> &crate::class::Class {
        &self.table.classes[self.nodes[tr.objs[i as usize] as usize].class]
    }

    /// Whether the node, a sink on a `plane_out` plane, is a write the runtime zeroes the plane for: an add or an or (their
    /// fold starts from zero), a sample sum or an ordered accumulate. The other updates fold into or replace the value the
    /// caller gave the plane, so it is kept.
    pub fn zeroes_plane(&self, node: u32, role: &str) -> Result<bool, Fault> {
        if role != "plane_out" {
            return Ok(false);
        }
        // an ordered accumulate is a refinement of the update that states no operation of its own, so it is asked first
        if self.is_a(node, "SampleSum") || self.is_a(node, "OrderedAccumWrite") {
            return Ok(true);
        }
        Ok(self.is_a(node, "Update")
            && matches!(self.str_field(node, "op")?.as_str(), "add" | "or"))
    }

    /// The total merge of the bindings, in visit order; a canonical node is the first of its key. Refuses two nodes that
    /// share a (role, name) key with different types.
    fn merge_bindings(&self, tr: &Traversal) -> Result<Bindings, Fault> {
        let (objs, w) = (&tr.objs, &tr.walk);
        let class_of = |i: u32| self.class_of_visit(tr, i);
        let mut held: Vec<(Key, Tt, String)> = vec![];
        let mut at: HashMap<Key, usize> = HashMap::new();
        let mut zeroed: HashSet<Key> = HashSet::new();
        let mut pairs: Vec<(u32, u32)> = vec![];
        for &v in &w.visits {
            let (node, canon) = (
                objs[v as usize],
                objs[w.order[w.pos[v as usize] as usize] as usize],
            );
            if class_of(v).binds && !self.is_scratch_leaf(node) && !self.is_scratch_sink(node) {
                pairs.push((node, canon))
            }
            if self.is_a(node, "Loop") {
                for sk in self.refs(node, "body_sinks")? {
                    pairs.push((sk, sk))
                }
            }
        }
        for (node, canon) in pairs {
            let key = (self.str_field(node, "role")?, self.str_field(node, "name")?);
            if self.zeroes_plane(node, &key.0)? {
                zeroed.insert(key.clone());
            }
            match at.get(&key) {
                None => {
                    at.insert(key.clone(), held.len());
                    held.push((
                        key,
                        self.tt(canon),
                        self.table.classes[self.nodes[canon as usize].class]
                            .name
                            .clone(),
                    ))
                }
                Some(&p) => {
                    let (a, b) = (&held[p].1, self.tt(node));
                    if *a != b {
                        return refuse(format!("leaf identity collision on (role={}, name={}): {} is typed {} but {} is typed {}. A slot is keyed on (role, name) ALONE, so these two \
would share one slot; the merge is TOTAL and the disagreement is refused, never silently collapsed.", rcore::keytext::py_str(&key.0), rcore::keytext::py_str(&key.1),
                            held[p].2, a.text(), self.table.classes[self.nodes[node as usize].class].name, b.text()));
                    }
                }
            }
        }
        Ok(Bindings { zeroed, held, at })
    }

    /// The quantities reached, by namespace, in walk order; refuses two quantities that share a namespace with different wires.
    fn reached_quantities(
        &self,
        tr: &Traversal,
        sep: &str,
    ) -> Result<Vec<(String, Quantity)>, Fault> {
        let (objs, w) = (&tr.objs, &tr.walk);
        let mut reached: Vec<(String, Quantity)> = vec![];
        for &i in &w.order {
            let h = objs[i as usize];
            if !self.is_a(h, "Leaf") {
                continue;
            }
            let Some(q) = Quantity::of(&self.field(h, "quantity")?, sep)? else {
                continue;
            };
            match reached.iter().find(|r| r.0 == q.prefix) {
                None => reached.push((q.prefix.clone(), q)),
                Some((_, held)) => {
                    if held.wires != q.wires {
                        let names = |q: &Quantity| {
                            q.wires
                                .iter()
                                .map(|x| rcore::keytext::py_str(&x.1))
                                .collect::<Vec<_>>()
                                .join(", ")
                        };
                        return refuse(format!("two quantities share the namespace {} but declare different wires ([{}] vs [{}]) - a namespace identifies a quantity ", rcore::keytext::py_str(&q.prefix), names(held), names(&q)));
                    }
                }
            }
        }
        Ok(reached)
    }

    /// The nodes of the walk, in order, that the access and dispatch passes watch: a bound slot's class, a declared
    /// class or a class the manifest watches.
    fn watched_nodes(&self, tr: &Traversal) -> Vec<u32> {
        let (objs, w) = (&tr.objs, &tr.walk);
        let class_of = |i: u32| self.class_of_visit(tr, i);
        let watched_of = |i: u32| {
            let c = class_of(i);
            c.watch && (c.decl.is_some() || c.spec.is_some_and(|s| s.watch == 1))
        };
        let watched: Vec<u32> = w
            .order
            .iter()
            .copied()
            .filter(|&i| watched_of(i))
            .map(|i| objs[i as usize])
            .collect();
        watched
    }

    /// The access class over the watched nodes (the highest rank any of them needs), raised to `force` when that is higher.
    fn access_class(
        &self,
        watched: &[u32],
        exempt: &[String],
        force: Option<&str>,
    ) -> Result<&'static str, Fault> {
        let rank_of = |a: &str| {
            rcore::facts::ACCESS_RANK
                .iter()
                .position(|x| *x == a)
                .unwrap_or(0)
        };
        let mut all = vec![];
        self.with_loop_bodies(watched, &mut all)?;
        let mut seam = HashSet::new();
        for &h in &all {
            if self.is_a(h, "Loop") {
                if let Some(x) = self.step_word(self.node_field_pub(h, "count")?, exempt)? {
                    seam.insert(x);
                }
            }
        }
        let mut cls = 0usize;
        for &h in &all {
            let param_cell = self.is_a(h, "At") && {
                let i = self.nodes[h as usize].ops[1];
                self.is_a(i, "RoleConst") && self.field(i, "label")? == s(str_const("PARAM_CELL"))
            };
            let scratch = self.is_scratch_sink(h)
                || (self.is_a(h, "At") && self.is_scratch_leaf(self.nodes[h as usize].ops[0]));
            if seam.contains(&h) || param_cell || scratch {
                continue;
            }
            let new = if self.is_a(h, "At") {
                1
            } else if self.is_a(h, "Update") && self.nodes[h as usize].ops.len() > 1 {
                2
            } else {
                0
            };
            cls = cls.max(new);
        }
        let mut access = rcore::facts::ACCESS_RANK[cls];
        if let Some(f) = force {
            if rank_of(f) > cls {
                access = rcore::facts::ACCESS_RANK[rank_of(f)]
            }
        }
        Ok(access)
    }

    /// The start value and the update of a loop carry: the loop that holds it names them by the carry's position.
    fn carry_sources(&self, carry: u32) -> Result<Vec<u32>, Fault> {
        for lp in (0..self.nodes.len() as u32).filter(|&n| self.is_a(n, "Loop")) {
            if let Some(at) = self.refs(lp, "carries")?.iter().position(|&c| c == carry) {
                let start = self.nodes[lp as usize].ops.get(at).copied();
                let update = self.refs(lp, "nexts")?.get(at).copied();
                return Ok(start.into_iter().chain(update).collect());
            }
        }
        Err(Fault::Refused(format!(
            "a loop carry {carry} that no loop holds"
        )))
    }

    /// The `(role, name)` slots the selector of the dispatch node `h` reads, sorted.
    ///
    /// The rule: a selector's reads are the plane leaves its closure reaches. A loop carry is a value of the loop, not a plane, so it
    /// contributes the leaves reached through its start value and through its update (transitively, through other carries, each
    /// node visited once), never a pair named after the carry. A gather's plane is a leaf its operands reach.
    pub fn selector_reads(&self, h: u32) -> Result<BTreeSet<(String, String)>, Fault> {
        let (mut seen, mut reads, mut st) =
            (HashSet::new(), BTreeSet::new(), vec![self.node(h)?.ops[0]]);
        while let Some(n) = st.pop() {
            if !seen.insert(n) {
                continue;
            }
            if self.is_a(n, "Leaf") {
                reads.insert((self.str_field(n, "role")?, self.str_field(n, "name")?));
            } else if self.is_a(n, "LoopCarry") {
                st.extend(self.carry_sources(n)?);
            }
            st.extend(self.nodes[n as usize].ops.iter().copied());
        }
        Ok(reads)
    }

    /// The dispatch listing: for each dispatch node its selector reads, and whether the selector is static (reads no
    /// plane the walk writes).
    fn dispatch_listing(&self, watched: &[u32]) -> Result<Listing, Fault> {
        let mut written: HashSet<String> = HashSet::new();
        for &h in watched {
            if self.is_a(h, "Sink") {
                written.insert(self.str_field(h, "name")?);
            }
            if self.is_a(h, "Loop") {
                for sk in self.refs(h, "body_sinks")? {
                    if self.is_a(sk, "Sink") {
                        written.insert(self.str_field(sk, "name")?);
                    }
                }
            }
        }
        let (mut listing, mut statics, mut all_reads) = (vec![], vec![], vec![]);
        for (i, &h) in self.dispatch_nodes(watched)?.iter().enumerate() {
            let reads = self.selector_reads(h)?;
            let stat = !reads.iter().any(|(_, nm)| written.contains(nm));
            statics.push(stat);
            all_reads.push(reads.iter().cloned().collect());
            listing.push(Val::Tuple(vec![
                s(&format!("dispatch{i}")),
                self.field(h, "K")?,
                Val::Tuple(reads.iter().map(|(a, b)| pair(a, b)).collect()),
                Val::Bool(stat),
            ]));
        }
        Ok((listing, statics, all_reads))
    }

    /// The names of the prior reads in the walk.
    fn prior_reads(&self, tr: &Traversal) -> Result<BTreeSet<String>, Fault> {
        let (objs, w) = (&tr.objs, &tr.walk);
        let mut prior = BTreeSet::new();
        for &i in &w.order {
            let h = objs[i as usize];
            if self.is_a(h, "Leaf") && self.str_field(h, "kind")? == "prior_read" {
                prior.insert(self.str_field(h, "name")?);
            }
        }
        Ok(prior)
    }

    /// The name of a loop's index node.
    fn node_idx_name(&self, lp: u32) -> Result<String, Fault> {
        match self.field(lp, "index")? {
            Val::Node(i) => self.str_field(i, "name"),
            _ => refuse("a loop's index is a node"),
        }
    }
    /// The node a field holds, or None when it holds none.
    fn node_field_pub(&self, h: u32, f: &str) -> Result<Option<u32>, Fault> {
        match self.field(h, f)? {
            Val::Node(n) => Ok(Some(n)),
            _ => Ok(None),
        }
    }

    /// The walk with handle `w`; refuses a handle the context does not hold.
    pub fn walk(&self, w: u32) -> Result<&WalkRes, Fault> {
        self.walks.get(w as usize).map_or_else(
            || refuse(format!("walk handle {w} is not in this context")),
            Ok,
        )
    }
}

/// The nodes of a walk's traversal, numbered by discovery, and the walk of their records.
struct Traversal {
    /// The node handle of each traversal index.
    objs: Vec<u32>,
    /// The canonical order, positions and visits over the records.
    walk: rcore::WalkOut,
}

/// The bound slots: each key's type and class name in order of first binding, and where each key sits.
struct Bindings {
    /// The written planes some sink of the walk zeroes.
    zeroed: HashSet<Key>,
    /// The key, type and class name of each slot.
    held: Vec<(Key, Tt, String)>,
    /// The index in `held` of each key.
    at: HashMap<Key, usize>,
}

/// Every slot in placement order, and which quantity wire (quantity index, wire index) owns each key.
struct Layout {
    /// The written planes the runtime zeroes (the walk's `zeroed`).
    zeroed: HashSet<Key>,
    /// The slots, in placement order.
    slots: Vec<(Key, Tt)>,
    /// The owning quantity and wire of each key a quantity expands to.
    owner: HashMap<Key, (usize, usize)>,
}

/// The exempt plane names: a tuple of texts, `()` when none; refuses another shape.
fn parse_exempt(exempt: Option<&str>) -> Result<Vec<String>, Fault> {
    let exempt: Vec<String> = match parse(exempt.unwrap_or("()")).map_err(Fault::Refused)? {
        Val::Tuple(v) | Val::List(v) => v
            .into_iter()
            .filter_map(|x| if let Val::Str(s) = x { Some(s) } else { None })
            .collect(),
        _ => return refuse("exempt planes: expected a tuple of str"),
    };
    Ok(exempt)
}

/// Adds the declared slots to the bound ones: a read is minted when the body binds none, a sink slot is kept for the
/// order and returned. Refuses a declared slot typed differently from the body's.
fn merge_declared(bound: &mut Bindings, declared: &Declared) -> Result<Vec<(Key, Tt)>, Fault> {
    let sink_roles = CONSTS
        .iter()
        .find(|c| c.0 == "SINK_ROLES")
        .map_or(&[][..], |c| c.1);
    let mut declared_sinks: Vec<(Key, Tt)> = vec![];
    bound.zeroed.extend(declared.zeroed.iter().cloned());
    let Bindings { held, at, .. } = bound;
    for (key, tt) in declared.slots.iter().cloned() {
        let (role, name) = key.clone();
        match at.get(&key) {
            None if sink_roles.contains(&role.as_str()) => declared_sinks.push((key, tt)),
            None => {
                at.insert(key.clone(), held.len());
                held.push((key, tt, "Leaf".into()))
            }
            Some(&p) => {
                if held[p].1 != tt {
                    return refuse(format!("declared slot (role={}, name={}) is typed {} but the body binds it as {}; a slot is keyed on (role, name) ALONE", rcore::keytext::py_str(&role), rcore::keytext::py_str(&name), tt.text(), held[p].1.text()));
                }
            }
        }
    }
    Ok(declared_sinks)
}

/// Places the slots: the quantities' wires first (a later wire of the same key replaces the earlier), then the declared
/// sink slots, then the bound ones. Refuses a namespaced key no quantity owns.
fn place_slots(
    reached: &[(String, Quantity)],
    declared_sinks: Vec<(Key, Tt)>,
    bound: &Bindings,
    sep: &str,
) -> Result<Layout, Fault> {
    let held = &bound.held;
    let mut owner: HashMap<Key, (usize, usize)> = HashMap::new();
    let mut slots: Vec<(Key, Tt)> = vec![];
    let mut pos_of: HashMap<Key, usize> = HashMap::new();
    for (qi, (_, q)) in reached.iter().enumerate() {
        for (i, s) in q.expand(sep).into_iter().enumerate() {
            owner.insert(s.0.clone(), (qi, i));
            match pos_of.get(&s.0) {
                Some(&p) => slots[p] = s,
                None => {
                    pos_of.insert(s.0.clone(), slots.len());
                    slots.push(s)
                }
            }
        }
    }
    for (key, tt) in declared_sinks {
        if !pos_of.contains_key(&key) {
            pos_of.insert(key.clone(), slots.len());
            slots.push((key, tt))
        }
    }
    for (key, tt, _) in held {
        if !pos_of.contains_key(key) {
            pos_of.insert(key.clone(), slots.len());
            slots.push((key.clone(), tt.clone()))
        }
    }
    if let Some(((role, name), _)) = slots
        .iter()
        .find(|(k, _)| k.1.contains(sep) && !owner.contains_key(k))
    {
        return refuse(format!("wire {} (role {}) is bound with no owning quantity: the {} namespace is reserved for compound-quantity wires, which only a Quantity declaration may mint ", rcore::keytext::py_str(name), rcore::keytext::py_str(role), rcore::keytext::py_str(sep)));
    }
    Ok(Layout {
        zeroed: bound.zeroed.clone(),
        slots,
        owner,
    })
}

/// The slots of one place of the role order: those a quantity owns, then the free ones.
type Placed<'a> = (Vec<&'a (Key, Tt)>, Vec<&'a (Key, Tt)>);

/// The slots in role order: quantity wires in declaration order, then the rest by name. Refuses a role outside the order.
///
/// The class a slot is placed by inside its role: the rank of a read plane's value, and for a written plane whether the runtime
/// leaves it as it is (1) or zeroes it (0), which `rcore::order_place` reads as a value plane and a scalar.
fn slot_class(key: &Key, tt: &Tt, zeroed: &HashSet<Key>) -> usize {
    if key.0 == "plane_out" {
        usize::from(!zeroed.contains(key))
    } else {
        tt.shape.len()
    }
}

/// A read plane is placed by the rank of its value, and a written plane by whether the runtime zeroes it (`slot_class`), so a
/// unit lists its planes with a value first, by rank, and its per-sample scalars after the wide planes.
fn order_slots(layout: &Layout) -> Result<Vec<(Key, Tt)>, Fault> {
    let (slots, owner) = (&layout.slots, &layout.owner);
    let roles = CONSTS
        .iter()
        .find(|c| c.0 == "ROLE_ORDER")
        .map_or(&[][..], |c| c.1);
    let mut places: BTreeMap<(usize, usize), Placed> = BTreeMap::new();
    for slot in slots {
        let (key, tt) = slot;
        let Some(place) = rcore::order_place(roles, &key.0, slot_class(key, tt, &layout.zeroed))
        else {
            return refuse(format!(
                "binding with a role outside the canonical order: {}",
                key.0
            ));
        };
        let (owned, free) = places.entry(place).or_default();
        if owner.contains_key(key) {
            owned.push(slot)
        } else {
            free.push(slot)
        }
    }
    let mut ordered: Vec<&(Key, Tt)> = vec![];
    for (mut owned, mut free) in places.into_values() {
        owned.sort_by_key(|(k, _)| owner[k]);
        free.sort_by(|a, b| (a.0).1.cmp(&(b.0).1));
        ordered.extend(owned);
        ordered.extend(free);
    }
    Ok(ordered.into_iter().cloned().collect())
}

/// A run of a quantity's slots: a role and, for a read plane, the rank class the role order separates it by.
type Run = (String, usize);

/// The spans: one consecutive, declaration-ordered run per role for each quantity, as text sorted by namespace. Refuses
/// a quantity whose slots are not one such run.
fn quantity_spans(
    reached: &[(String, Quantity)],
    ordered: &[(Key, Tt)],
    zeroed: &HashSet<Key>,
    sep: &str,
) -> Result<Spans, Fault> {
    let slot_of: HashMap<&Key, usize> = ordered
        .iter()
        .enumerate()
        .map(|(i, (k, _))| (k, i))
        .collect();
    let order = CONSTS
        .iter()
        .find(|c| c.0 == "ROLE_ORDER")
        .map_or(&[][..], |c| c.1);
    // the spans: one consecutive, declaration-ordered run per role
    let mut spans: Vec<(String, String)> = vec![];
    let mut rows: Vec<QuantityRow> = vec![];
    for (prefix, q) in reached {
        let expanded = q.expand(sep);
        let (mut idx, mut per): (Vec<i128>, Vec<(Run, Vec<i128>)>) = (vec![], vec![]);
        for (k, tt) in &expanded {
            let i = slot_of[k] as i128;
            idx.push(i);
            // a run is one role and, for a read plane, one rank class: the order lists them apart
            let class =
                rcore::order_place(order, &k.0, slot_class(k, tt, zeroed)).map_or(0, |p| p.1);
            match per.iter_mut().find(|p| p.0 == (k.0.clone(), class)) {
                Some(p) => p.1.push(i),
                None => per.push(((k.0.clone(), class), vec![i])),
            }
        }
        for ((role, _), run) in &per {
            let mut sorted = run.clone();
            sorted.sort();
            if *run != sorted
                || run
                    .iter()
                    .enumerate()
                    .any(|(j, v)| *v != run[0] + j as i128)
            {
                return refuse(format!("quantity {} occupies a non-contiguous or mis-ordered span in role {}: arg_spec indices {:?} (require one consecutive, declaration-ordered run per role)", rcore::keytext::py_str(prefix), rcore::keytext::py_str(role), run));
            }
        }
        let span = Val::Rec(
            "QuantitySpan".into(),
            vec![
                ("prefix".into(), s(prefix)),
                (
                    "wires".into(),
                    Val::Tuple(expanded.iter().map(|(k, _)| s(&k.1)).collect()),
                ),
                (
                    "slots".into(),
                    Val::Tuple(idx.iter().map(|&i| Val::Int(i)).collect()),
                ),
            ],
        );
        rows.push((
            prefix.clone(),
            expanded.iter().map(|(k, _)| k.1.clone()).collect(),
            idx.iter().map(|&i| i as usize).collect(),
        ));
        spans.push((prefix.to_string(), text_val(&span)));
    }
    spans.sort();
    Ok((spans, rows))
}

/// What a digest takes beside the slots: the node key lines, and the role of each slot as the digest names it.
struct DigestKeys<'a> {
    /// The node key lines, one per node, in walk order.
    pre: &'a str,
    /// The role of each ordered slot as the digest takes it.
    roles: &'a [String],
}

/// The digest tail and the walk's fields as text, and the digest.
fn render_fields(
    ordered: &[(Key, Tt)],
    spans: &[(String, String)],
    access: &str,
    keys: DigestKeys,
    listing: Vec<Val>,
    prior: &BTreeSet<String>,
    finish: &Val,
) -> (Vec<(&'static str, String)>, String) {
    let DigestKeys {
        pre,
        roles: taken_roles,
    } = keys;
    let (mut tail, mut leaves, mut spec, mut types, mut index) =
        (String::new(), vec![], vec![], vec![], vec![]);
    let mut tail_spec = vec![];
    for (i, ((role, name), tt)) in ordered.iter().map(|x| (&x.0, &x.1)).enumerate() {
        let taken = &taken_roles[i];
        tail.push_str(&format!("leaf|{taken}|{name}|{}\n", tt.slot_text()));
        tail_spec.push(pair(taken, name));
        leaves.push(Val::Rec(
            "Slot".into(),
            vec![
                ("role".into(), s(role)),
                ("name".into(), s(name)),
                ("ttype".into(), tt.val()),
            ],
        ));
        spec.push(pair(role, name));
        types.push(Val::Tuple(vec![pair(role, name), tt.val()]));
        index.push(Val::Tuple(vec![pair(role, name), Val::Int(i as i128)]));
    }
    tail.push_str(&format!(
        "arg_spec|{}\naccess|{access}|None\n",
        text_val(&Val::Tuple(tail_spec))
    ));
    for (_, q) in spans {
        tail.push_str(&format!("quantity|{q}\n"))
    }
    let digest = rcore::digest(pre, tail.as_bytes());
    let t = |v: Vec<Val>| text_val(&Val::Tuple(v));
    let fields = vec![
        ("leaves", t(leaves)),
        ("arg_spec", t(spec)),
        ("slot_of", t(index)),
        ("slot_types", t(types)),
        ("access", text_val(&s(access))),
        (
            "quantities",
            format!(
                "({}{})",
                spans
                    .iter()
                    .map(|(p, q)| format!("({}, {q})", rcore::keytext::py_str(p)))
                    .collect::<Vec<_>>()
                    .join(", "),
                if spans.len() == 1 { "," } else { "" }
            ),
        ),
        ("digest", text_val(&s(&digest))),
        ("dispatches", t(listing)),
        ("prior", t(prior.iter().map(|p| s(p)).collect())),
        ("finish", text_val(finish)),
        ("preimage", pre.to_string()),
    ];
    (fields, digest)
}

/// An optional C string argument: None for NULL.
fn opt<'a>(p: *const c_char, what: &str) -> Result<Option<&'a str>, Fault> {
    if p.is_null() {
        Ok(None)
    } else {
        cstr(p, what).map(Some)
    }
}

/// A door function over raw pointers.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_canonical(
    ctx: *mut Ctx,
    sinks: *const u32,
    n: usize,
    declared_kt: *const c_char,
    force_access: *const c_char,
    exempt_kt: *const c_char,
    out: *mut u32,
) -> c_int {
    guard(ctx, |c| {
        if out.is_null() {
            return badarg("out is NULL");
        }
        let w = c.canonical(
            slice(sinks, n, "sinks")?,
            &declared_slots(opt(declared_kt, "declared_kt")?)?,
            opt(force_access, "force_access")?,
            &parse_exempt(opt(exempt_kt, "exempt_kt")?)?,
        )?;
        unsafe { *out = w as u32 }
        Ok(())
    })
}

/// The canonical walk over `sinks` that keeps the slots the walk `like` binds, so a sub-render of a
/// unit binds the same arguments as its entry; no plane is exempt. Refuses a walk the context does
/// not hold.
///
/// # Safety
/// `sinks` is valid for `n` nodes and `out` for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_canonical_like(
    ctx: *mut Ctx,
    sinks: *const u32,
    n: usize,
    like: u32,
    out: *mut u32,
) -> c_int {
    guard(ctx, |c| {
        if out.is_null() {
            return badarg("out is NULL");
        }
        let held = c.walk(like)?;
        let kept = Declared {
            slots: held.leaves.clone(),
            zeroed: held.zeroed.clone(),
            counts: held.counts.clone(),
            winners: held.winners.clone(),
        };
        let w = c.canonical(slice(sinks, n, "sinks")?, &kept, None, &[])?;
        unsafe { *out = w as u32 }
        Ok(())
    })
}

/// A door function over raw pointers.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_walk_order(
    ctx: *mut Ctx,
    w: u32,
    buf: *mut u32,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        if n.is_null() {
            return badarg("n is NULL");
        }
        let o = &c.walk(w)?.order;
        unsafe { *n = o.len() }
        if cap < o.len() || (buf.is_null() && !o.is_empty()) {
            return Err(Fault::Truncated);
        }
        unsafe { std::ptr::copy_nonoverlapping(o.as_ptr(), buf, o.len()) }
        Ok(())
    })
}

/// The text of one field of a walk, through the two-call text protocol. Not exported: only the
/// replay tools in rs/support read the walk's fields as text, to compare them with recorded ones;
/// a backend asks the typed answers (`rx_walk_binds`, `rx_walk_args`, `rx_canonical_like`).
///
/// # Safety
/// `field` is a NUL-terminated string; `buf` is valid for `cap` bytes and `len` for one write.
pub unsafe extern "C" fn rx_walk_field(
    ctx: *mut Ctx,
    w: u32,
    field: *const c_char,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let f = cstr(field, "field")?;
        let t = c
            .walk(w)?
            .fields
            .iter()
            .find(|x| x.0 == f)
            .map(|x| x.1.clone());
        put(
            buf,
            cap,
            len,
            &t.map_or_else(|| refuse(format!("a walk has no field {f:?}")), Ok)?,
        )
    })
}

/// A door function over raw pointers.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_digest(ctx: *mut Ctx, w: u32, out65: *mut c_char) -> c_int {
    guard(ctx, |c| {
        if out65.is_null() {
            return badarg("out65 is NULL");
        }
        let d = &c.walk(w)?.digest;
        unsafe {
            std::ptr::copy_nonoverlapping(d.as_ptr(), out65 as *mut u8, d.len());
            *out65.add(d.len()) = 0
        }
        Ok(())
    })
}

/// A door function over raw pointers.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_walk_position(
    ctx: *mut Ctx,
    w: u32,
    node: u32,
    pos: *mut u32,
) -> c_int {
    guard(ctx, |c| {
        if pos.is_null() {
            return badarg("pos is NULL");
        }
        c.node(node)?;
        match c.walk(w)?.positions.get(&node) {
            Some(&p) => { unsafe { *pos = p }; Ok(()) }
            None => refuse(format!("node {node} is not reached by the walk's traversal; a node inside a loop body is not in the flat walk (see rx_body_nodes)")),
        }
    })
}
