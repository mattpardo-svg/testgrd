"""Count-only diagnostic for CGM limits, boundary lines and NCP assessed-element links.

Prints counts and CIM class/property names only, never ids, names or values.
Usage: python diag_counts.py <cgm zip> <ncp zip>
"""
import collections
import io
import os
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
LIMIT_CLASSES = {"CurrentLimit", "ApparentPowerLimit", "ActivePowerLimit", "VoltageLimit"}


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def normalise_id(reference):
    reference = reference.rsplit("#", 1)[-1]
    if reference.startswith("urn:uuid:"):
        reference = reference[9:]
    return reference.lstrip("_").lower()


def element_id(element):
    value = element.get(RDF + "ID") or element.get(RDF + "about")
    return normalise_id(value) if value else None


def xml_files(source):
    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            lower = name.lower()
            if lower.endswith(".zip"):
                yield from xml_files(io.BytesIO(data))
            elif lower.endswith(".xml"):
                yield name, data


def file_kind(root):
    profiles, keyword = [], None
    for element in root:
        if local_name(element.tag) != "FullModel":
            continue
        for child in element:
            text = (child.text or child.get(RDF + "resource") or "").strip()
            if local_name(child.tag) == "Model.profile":
                profiles.append(text)
            elif local_name(child.tag) == "conformsTo":
                profiles.append(text)
            elif local_name(child.tag) == "keyword":
                keyword = text
    joined = " ".join(profiles)
    if "EquipmentBoundary" in joined:
        return "EQBD"
    if "EquipmentCore" in joined or "CoreEquipment" in joined:
        return "EQ"
    if keyword:
        return keyword
    for marker, kind in (("SteadyStateHypothesis", "SSH"), ("Topology", "TP"), ("StateVariables", "SV")):
        if marker in joined:
            return kind
    return "other"


def section(title):
    print()
    print("=== " + title + " ===")


def print_counter(counter):
    if not counter:
        print("  (none)")
    for key, count in sorted(counter.items(), key=lambda item: str(item[0])):
        print("  {:<70} {}".format(" | ".join(key) if isinstance(key, tuple) else str(key), count))


def analyse_cgm(cgm_zip):
    class_of = {}
    terminal_equipment = {}
    set_terminal, set_equipment = {}, {}
    limit_info = {}
    type_kind = {}
    set_link_counter = collections.Counter()
    kinds = collections.Counter()
    all_xml = []

    for name, data in xml_files(cgm_zip):
        all_xml.append((name, data))
        root = ET.fromstring(data)
        kind = file_kind(root)
        kinds[kind] += 1
        if kind not in ("EQ", "EQBD"):
            continue
        for element in root:
            identifier = element_id(element)
            cls = local_name(element.tag)
            if identifier and cls != "FullModel":
                class_of.setdefault(identifier, cls)
            refs = {}
            texts = {}
            for child in element:
                resource = child.get(RDF + "resource")
                if resource:
                    refs[local_name(child.tag)] = normalise_id(resource)
                else:
                    texts[local_name(child.tag)] = (child.text or "").strip()
            if cls == "Terminal" and "Terminal.ConductingEquipment" in refs:
                terminal_equipment[identifier] = refs["Terminal.ConductingEquipment"]
            elif cls == "OperationalLimitSet":
                has_terminal = "OperationalLimitSet.Terminal" in refs
                has_equipment = "OperationalLimitSet.Equipment" in refs
                set_link_counter[("Terminal" if has_terminal else "-", "Equipment" if has_equipment else "-")] += 1
                if has_terminal:
                    set_terminal[identifier] = refs["OperationalLimitSet.Terminal"]
                if has_equipment:
                    set_equipment[identifier] = refs["OperationalLimitSet.Equipment"]
            elif cls in LIMIT_CLASSES:
                limit_info[identifier] = (cls, refs.get("OperationalLimit.OperationalLimitSet"),
                                          refs.get("OperationalLimit.OperationalLimitType"))
            elif cls == "OperationalLimitType":
                kind_value = (refs.get("OperationalLimitType.limitType") or texts.get("OperationalLimitType.limitType")
                              or refs.get("OperationalLimitType.kind") or texts.get("OperationalLimitType.kind") or "?")
                has_duration = "OperationalLimitType.acceptableDuration" in texts
                type_kind[identifier] = kind_value.rsplit(".", 1)[-1] + ("+duration" if has_duration else "")

    def equipment_class_of_set(set_id):
        equipment = set_equipment.get(set_id) or terminal_equipment.get(set_terminal.get(set_id))
        return class_of.get(equipment, "unresolved") if equipment else "unresolved"

    limit_counter = collections.Counter()
    for limit_class, set_id, type_id in limit_info.values():
        limit_counter[(limit_class, type_kind.get(type_id, "no type"), equipment_class_of_set(set_id))] += 1

    section("CGM files by kind")
    print_counter(kinds)
    section("EQ OperationalLimitSet links (Terminal | Equipment)")
    print_counter(set_link_counter)
    section("EQ limits by class | limit type | equipment class")
    print_counter(limit_counter)

    pypowsybl_counts(all_xml)
    return class_of, limit_info, set_terminal, set_equipment, terminal_equipment


def pypowsybl_counts(all_xml):
    section("pypowsybl")
    try:
        import pypowsybl as pp
        folder = tempfile.mkdtemp()
        flat_zip = os.path.join(folder, "flat_cgm.zip")
        with zipfile.ZipFile(flat_zip, "w") as out:
            used = set()
            for index, (name, data) in enumerate(all_xml):
                base = os.path.basename(name)
                if base in used:
                    base = "{}_{}".format(index, base)
                used.add(base)
                out.writestr(base, data)
        network = pp.network.load(flat_zip)
        limits = network.get_operational_limits().reset_index()
        print("  limits by element_type | type (all):")
        print_counter(collections.Counter(zip(limits["element_type"].astype(str), limits["type"].astype(str))))
        permanent = limits[limits["acceptable_duration"] == -1]
        print("  permanent limits (PATL) by element_type | type:")
        print_counter(collections.Counter(zip(permanent["element_type"].astype(str), permanent["type"].astype(str))))
        transformers = network.get_2_windings_transformers()
        with_limits = set(limits.loc[limits["element_type"] == "TWO_WINDINGS_TRANSFORMER", "element_id"])
        print("  2W transformers: {} total, {} with any limit".format(len(transformers), len(with_limits)))
        dangling = network.get_dangling_lines()
        connected = int(dangling["connected"].sum()) if "connected" in dangling else -1
        print("  dangling lines: {} total, {} connected".format(len(dangling), connected))
        pp.loadflow.run_ac(network)
        dangling = network.get_dangling_lines()
        if "i" in dangling and "connected" in dangling:
            on = dangling[dangling["connected"]]
            print("  after AC load flow: connected dangling lines with NaN current: {} of {}".format(
                int(on["i"].isna().sum()), len(on)))
    except Exception as error:  # message may contain ids, so print the type only
        print("  pypowsybl step failed: " + type(error).__name__)


def analyse_ncp(ncp_zip, class_of, limit_info, set_terminal, set_equipment, terminal_equipment):
    tag_counter = collections.Counter()
    value_counter = collections.Counter()
    ae_reference_counter = collections.Counter()
    watched = ("AssessedElement", "AssessedElementWithContingency", "AssessedElementWithRemedialAction",
               "ContingencyWithRemedialAction", "Contingency", "OrdinaryContingency", "ExceptionalContingency")

    def equipment_class(reference):
        cls = class_of.get(reference)
        if cls in LIMIT_CLASSES:
            set_id = limit_info.get(reference, (None, None, None))[1]
            equipment = set_equipment.get(set_id) or terminal_equipment.get(set_terminal.get(set_id))
            return cls + "->" + class_of.get(equipment, "unresolved")
        return cls or "not in EQ"

    for _name, data in xml_files(ncp_zip):
        root = ET.fromstring(data)
        kind = file_kind(root)
        for element in root:
            cls = local_name(element.tag)
            if cls in watched:
                tag_counter[(kind, cls)] += 1
            for child in element:
                prop = local_name(child.tag)
                resource = child.get(RDF + "resource")
                if prop.startswith(("AssessedElementWithContingency.", "AssessedElementWithRemedialAction.",
                                    "ContingencyWithRemedialAction.")):
                    tag_counter[(kind, prop)] += 1
                if prop in ("AssessedElement.isCombinableWithContingency",
                            "AssessedElement.isCombinableWithRemedialAction",
                            "AssessedElementWithContingency.combinationConstraintKind",
                            "AssessedElementWithContingency.normalEnabled"):
                    value = (resource or child.text or "").strip().rsplit(".", 1)[-1].rsplit("#", 1)[-1]
                    value_counter[(kind, prop, value)] += 1
                if cls == "AssessedElement" and resource:
                    ae_reference_counter[(prop, equipment_class(normalise_id(resource)))] += 1

    section("NCP classes and link properties (file kind | name)")
    print_counter(tag_counter)
    section("NCP switch values (file kind | property | value)")
    print_counter(value_counter)
    section("AssessedElement references (property | referenced EQ class)")
    print_counter(ae_reference_counter)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    eq_maps = analyse_cgm(sys.argv[1])
    analyse_ncp(sys.argv[2], *eq_maps)


if __name__ == "__main__":
    main()
