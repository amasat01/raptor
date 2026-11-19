//! What a loader says of a manifest's schema version and execution axis. Oracle: the texts written out below, which a caller
//! (and a user reading a refusal) matches on; the supported version is 3 and the two ABI tags are `aether-abi/1` and `aether-abi/2`.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a panic is the failure
use rcore::manifest::{check_execution_axis, check_schema_version};
use rcore::pjson::Json;

/// An object of the given string and integer members, in the order written.
fn doc(members: &[(&str, Json)]) -> Json {
    Json::Obj(
        members
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect(),
    )
}

/// A string value.
fn text(s: &str) -> Json {
    Json::Str(s.into())
}

#[test]
fn a_manifest_above_the_supported_version_is_refused_by_version_asking_for_an_upgrade() {
    let newer = doc(&[("schema_version", Json::Int(4))]);
    assert_eq!(
        check_schema_version(&newer, "manifest.json", false).unwrap_err(),
        "ValueError: manifest.json was built for plugin schema v4, but this loader supports v3; upgrade eagle to load it"
    );
}

#[test]
fn the_supported_version_loads_and_a_manifest_without_a_version_is_version_one() {
    let current = doc(&[("schema_version", Json::Int(3))]);
    assert_eq!(check_schema_version(&current, "m", false), Ok(3));
    assert_eq!(check_schema_version(&doc(&[]), "m", false), Ok(1));
}

#[test]
fn the_older_version_key_is_read_only_when_the_legacy_reading_is_asked() {
    let old = doc(&[("version", Json::Int(2))]);
    assert_eq!(check_schema_version(&old, "m", true), Ok(2));
    assert_eq!(check_schema_version(&old, "m", false), Ok(1));
}

#[test]
fn a_version_one_manifest_with_the_version_two_tag_is_refused_naming_the_tag_it_requires() {
    let tagged = doc(&[("aether_abi", text("aether-abi/2"))]);
    assert_eq!(
        check_execution_axis(&tagged, 1, "manifest.json").unwrap_err(),
        "ValueError: manifest.json: schema v1 requires aether_abi='aether-abi/1' when present; got 'aether-abi/2'"
    );
}

#[test]
fn a_version_one_manifest_with_its_own_tag_is_refused_for_carrying_execution_keys() {
    let downgraded = doc(&[
        ("aether_abi", text("aether-abi/1")),
        ("exec_targets", Json::Arr(vec![text("host")])),
        ("exec_access", text("sample_local")),
    ]);
    let refused = check_execution_axis(&downgraded, 1, "manifest.json").unwrap_err();
    assert!(
        refused.contains("schema v1 must not carry execution key(s)")
            && refused.contains("exec_access")
            && refused.contains("exec_targets"),
        "{refused}"
    );
}

#[test]
fn a_version_one_manifest_with_no_tag_and_no_execution_key_passes_the_axis() {
    assert_eq!(check_execution_axis(&doc(&[]), 1, "m"), Ok(()));
}

#[test]
fn a_version_two_manifest_without_its_execution_keys_is_refused_naming_them() {
    let bare = doc(&[("aether_abi", text("aether-abi/2"))]);
    let refused = check_execution_axis(&bare, 2, "manifest.json").unwrap_err();
    assert!(
        refused.contains("schema v2 requires execution key(s)")
            && refused.contains("exec_targets")
            && refused.contains("exec_access"),
        "{refused}"
    );
}

/// A parsed document as the validator's value: numbers that are whole are integers, as a manifest writes them.
fn value(j: &rcore::json::J) -> Json {
    use rcore::json::J;
    match j {
        J::Null => Json::Null,
        J::Bool(b) => Json::Bool(*b),
        #[allow(clippy::cast_possible_truncation, clippy::float_cmp)]
        J::Num(n) if n.fract() == 0.0 => Json::Int(*n as i128),
        J::Num(n) => Json::Float(*n),
        J::Str(s) => Json::Str(s.clone()),
        J::Arr(items) => Json::Arr(items.iter().map(value).collect()),
        J::Obj(members) => Json::Obj(members.iter().map(|(k, v)| (k.clone(), value(v))).collect()),
    }
}

/// The manifest a host build of one kernel publishes, as the front end writes it.
const WRITTEN: &str = include_str!("data/hawk_host_manifest.json");

/// The vocabulary the device wing's manifest states, which the validator checks a document against.
fn vocabulary() -> rcore::manifest::Vocab {
    let wing = rcore::json::parse(include_str!("../../wings/manifests/simt.json")).unwrap();
    rcore::manifest::Vocab::from_json(&value(wing.get("vocabulary").unwrap())).unwrap()
}

#[test]
fn the_manifest_a_front_end_writes_passes_the_shape_validator() {
    let document = value(&rcore::json::parse(WRITTEN).unwrap());
    assert_eq!(rcore::manifest::validate(&document, &vocabulary()), Ok(()));
}

#[test]
fn the_same_manifest_with_its_top_level_keys_in_another_order_is_refused() {
    let Json::Obj(mut members) = value(&rcore::json::parse(WRITTEN).unwrap()) else {
        panic!("the manifest is an object");
    };
    members.reverse();
    let refused = rcore::manifest::validate(&Json::Obj(members), &vocabulary()).unwrap_err();
    assert!(refused.contains("order"), "{refused}");
}
