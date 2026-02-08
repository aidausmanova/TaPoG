import xml.etree.ElementTree as ET
import json
import uuid
import os
from pathlib import Path
from collections import defaultdict

# XML Namespaces used in XBRL files
NAMESPACES = {
    'link': 'http://www.xbrl.org/2003/linkbase',
    'xlink': 'http://www.w3.org/1999/xlink',
    'xsd': 'http://www.w3.org/2001/XMLSchema',
    'label': 'http://www.xbrl.org/2003/linkbase',
    'ref': 'http://www.xbrl.org/2003/linkbase',
    'gen': 'http://xbrl.org/2008/generic',
    'xbrldt': 'http://xbrl.org/2005/xbrldt'
}


def extract_element_name_from_href(href):
    """Extract element name from href attribute."""
    if not href or '#' not in href:
        return None
    # Handle both formats: ../file.xsd#element and file.xsd#element
    return href.split('#')[1] if '#' in href else None


def parse_xsd_file(xsd_path):
    """
    Parse XSD schema file to extract element definitions and role types.
    Returns dict of elements and roles.
    """
    tree = ET.parse(xsd_path)
    root = tree.getroot()
    
    elements = {}
    roles = {}
    
    # Extract role types
    for role_type in root.findall('.//link:roleType', NAMESPACES):
        role_id = role_type.get('id')
        role_uri = role_type.get('roleURI')
        definition_elem = role_type.find('link:definition', NAMESPACES)
        definition = definition_elem.text if definition_elem is not None else ''
        
        roles[role_uri] = {
            'id': role_id,
            'definition': definition
        }
    
    # Extract elements from xsd:element tags
    for element in root.findall('.//{http://www.w3.org/2001/XMLSchema}element'):
        name = element.get('name')
        elem_id = element.get('id')
        elem_type = element.get('type')
        abstract = element.get('abstract', 'false')
        
        if name:
            elements[name] = {
                'id': elem_id,
                'type': elem_type,
                'abstract': abstract == 'true',
                'name': name
            }
    
    return elements, roles


def parse_label_file(label_path):
    """
    Parse label linkbase file to extract labels for elements.
    Returns dict mapping element name to labels.
    """
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        return {}
    
    tree = ET.parse(label_path)
    root = tree.getroot()
    
    labels = defaultdict(list)
    
    # Process each labelLink
    for label_link in root.findall('.//link:labelLink', NAMESPACES):
        # Build mapping of locators within this link
        locators = {}
        for loc in label_link.findall('.//link:loc', NAMESPACES):
            label = loc.get('{http://www.w3.org/1999/xlink}label')
            href = loc.get('{http://www.w3.org/1999/xlink}href')
            element_name = extract_element_name_from_href(href)
            if element_name:
                locators[label] = element_name
        
        # Build mapping of label resources
        label_resources = {}
        for label_elem in label_link.findall('.//link:label', NAMESPACES):
            label_id = label_elem.get('{http://www.w3.org/1999/xlink}label')
            role = label_elem.get('{http://www.w3.org/1999/xlink}role', '')
            text = label_elem.text or ''
            lang = label_elem.get('{http://www.w3.org/XML/1998/namespace}lang', 'en')
            
            label_resources[label_id] = {
                'text': text.strip(),
                'role': role,
                'lang': lang
            }
        
        # Connect labels to elements via arcs
        for arc in label_link.findall('.//link:labelArc', NAMESPACES):
            from_label = arc.get('{http://www.w3.org/1999/xlink}from')
            to_label = arc.get('{http://www.w3.org/1999/xlink}to')
            
            if from_label in locators and to_label in label_resources:
                element_name = locators[from_label]
                label_info = label_resources[to_label]
                labels[element_name].append(label_info)
    
    return dict(labels)


def parse_documentation_file(doc_path):
    """
    Parse documentation linkbase file to extract documentation for elements.
    Returns dict mapping element name to documentation.
    """
    if not os.path.exists(doc_path):
        print(f"Warning: Documentation file not found: {doc_path}")
        return {}
    
    tree = ET.parse(doc_path)
    root = tree.getroot()
    
    documentation = {}
    
    # Process each labelLink (documentation uses labelLink structure)
    for label_link in root.findall('.//link:labelLink', NAMESPACES):
        # Build mapping of locators
        locators = {}
        for loc in label_link.findall('.//link:loc', NAMESPACES):
            label = loc.get('{http://www.w3.org/1999/xlink}label')
            href = loc.get('{http://www.w3.org/1999/xlink}href')
            element_name = extract_element_name_from_href(href)
            if element_name:
                locators[label] = element_name
        
        # Build mapping of documentation resources
        doc_resources = {}
        for label_elem in label_link.findall('.//link:label', NAMESPACES):
            label_id = label_elem.get('{http://www.w3.org/1999/xlink}label')
            role = label_elem.get('{http://www.w3.org/1999/xlink}role', '')
            text = label_elem.text or ''
            
            # Check if this is a documentation label
            if 'documentation' in role.lower():
                doc_resources[label_id] = text.strip()
        
        # Connect documentation to elements via arcs
        for arc in label_link.findall('.//link:labelArc', NAMESPACES):
            from_label = arc.get('{http://www.w3.org/1999/xlink}from')
            to_label = arc.get('{http://www.w3.org/1999/xlink}to')
            
            if from_label in locators and to_label in doc_resources:
                element_name = locators[from_label]
                if doc_resources[to_label]:  # Only add non-empty documentation
                    documentation[element_name] = doc_resources[to_label]
    
    return documentation


def parse_presentation_file(pre_path):
    """
    Parse presentation linkbase file to extract hierarchical relationships.
    Returns list of parent-child relationships.
    """
    if not os.path.exists(pre_path):
        return []
    
    tree = ET.parse(pre_path)
    root = tree.getroot()
    
    relationships = []
    
    # Process each presentation link
    for pres_link in root.findall('.//link:presentationLink', NAMESPACES):
        role = pres_link.get('{http://www.w3.org/1999/xlink}role')
        
        # Build mapping of locators within this link
        locators = {}
        for loc in pres_link.findall('.//link:loc', NAMESPACES):
            label = loc.get('{http://www.w3.org/1999/xlink}label')
            href = loc.get('{http://www.w3.org/1999/xlink}href')
            element_name = extract_element_name_from_href(href)
            if element_name:
                locators[label] = element_name
        
        # Extract parent-child relationships from arcs
        for arc in pres_link.findall('.//link:presentationArc', NAMESPACES):
            from_label = arc.get('{http://www.w3.org/1999/xlink}from')
            to_label = arc.get('{http://www.w3.org/1999/xlink}to')
            order = arc.get('order', '0')
            
            if from_label in locators and to_label in locators:
                parent = locators[from_label]
                child = locators[to_label]
                relationships.append({
                    'parent': parent,
                    'child': child,
                    'order': float(order),
                    'role': role
                })
    
    return relationships


def parse_reference_file(ref_path):
    """
    Parse reference linkbase file to extract references for elements.
    Returns dict mapping element name to references.
    """
    if not os.path.exists(ref_path):
        return {}
    
    tree = ET.parse(ref_path)
    root = tree.getroot()
    
    references = defaultdict(list)
    
    # Process each referenceLink
    for ref_link in root.findall('.//link:referenceLink', NAMESPACES):
        # Build mapping of locators
        locators = {}
        for loc in ref_link.findall('.//link:loc', NAMESPACES):
            label = loc.get('{http://www.w3.org/1999/xlink}label')
            href = loc.get('{http://www.w3.org/1999/xlink}href')
            element_name = extract_element_name_from_href(href)
            if element_name:
                locators[label] = element_name
        
        # Build mapping of reference resources
        ref_resources = {}
        for ref_elem in ref_link.findall('.//link:reference', NAMESPACES):
            ref_id = ref_elem.get('{http://www.w3.org/1999/xlink}label')
            ref_parts = {}
            
            for child in ref_elem:
                tag = child.tag.split('}')[1] if '}' in child.tag else child.tag
                if child.text:
                    ref_parts[tag] = child.text.strip()
            
            if ref_parts:
                ref_resources[ref_id] = ref_parts
        
        # Connect references to elements via arcs
        for arc in ref_link.findall('.//link:referenceArc', NAMESPACES):
            from_label = arc.get('{http://www.w3.org/1999/xlink}from')
            to_label = arc.get('{http://www.w3.org/1999/xlink}to')
            
            if from_label in locators and to_label in ref_resources:
                element_name = locators[from_label]
                references[element_name].append(ref_resources[to_label])
    
    return dict(references)


def build_hierarchy(relationships):
    """
    Build hierarchical structure from parent-child relationships.
    Returns parent->children map and child->parents map.
    """
    children_map = defaultdict(list)
    parent_map = defaultdict(set)
    all_elements = set()
    
    for rel in relationships:
        parent = rel['parent']
        child = rel['child']
        
        children_map[parent].append({
            'name': child,
            'order': rel['order']
        })
        parent_map[child].add(parent)
        all_elements.add(parent)
        all_elements.add(child)
    
    # Sort children by order
    for parent in children_map:
        children_map[parent].sort(key=lambda x: x['order'])
    
    # Find root elements (elements that are parents but not children)
    all_parents = set(children_map.keys())
    all_children = set(parent_map.keys())
    roots = all_parents - all_children
    
    return dict(children_map), dict(parent_map), list(roots), all_elements


def get_path_to_root(element_name, parent_map, labels, max_depth=20):
    """
    Get the path from element to root, returning list of (name, label) tuples.
    Returns path from immediate parent to root (excludes the element itself).
    """
    path_names = []
    path_labels = []
    visited = set()
    current = element_name
    
    # Start from the element's parents
    if current not in parent_map:
        return [], []
    
    depth = 0
    # Traverse up the hierarchy
    while current in parent_map and depth < max_depth:
        if current in visited:  # Avoid circular references
            break
        visited.add(current)
        
        parents = list(parent_map[current])
        if not parents:
            break
        
        # Take the first parent (or we could handle multiple parents differently)
        parent = parents[0]
        
        # Get label for parent
        parent_label = get_element_label(parent, labels)
        
        path_names.append(parent)
        path_labels.append(parent_label)
        
        current = parent
        depth += 1
    
    # Reverse to get root-to-element order
    path_names.reverse()
    path_labels.reverse()
    
    return path_names, path_labels


def get_element_label(element_name, labels):
    """
    Get the primary label for an element.
    """
    element_labels = labels.get(element_name, [])
    
    # Look for standard label first
    for lbl in element_labels:
        role = lbl.get('role', '')
        if '/role/label' in role and 'verbose' not in role and 'terse' not in role:
            return lbl['text']
    
    # Fallback to any label
    if element_labels:
        return element_labels[0]['text']
    
    # Last resort: format the element name
    return element_name.replace('_', ' ').replace('-', ' ').title()


def is_leaf_element(element_name, children_map):
    """
    Check if an element is a leaf (has no children).
    """
    return element_name not in children_map or len(children_map[element_name]) == 0


def convert_taxonomy_to_json(taxonomy_root_path, output_path):
    """
    Main function to convert XBRL taxonomy to JSON format.
    """
    taxonomy_root = Path(taxonomy_root_path)
    
    # Locate files
    xsd_files = list(taxonomy_root.glob('*-cor_*.xsd'))
    if not xsd_files:
        print("Error: No XSD schema file found!")
        return {}
    xsd_file = xsd_files[0]
    
    label_files = list(taxonomy_root.glob('labels/lab_*-en_*.xml'))
    label_file = label_files[0] if label_files else None
    
    doc_files = list(taxonomy_root.glob('labels/doc_*-en_*.xml'))
    doc_file = doc_files[0] if doc_files else None
    
    # Find all presentation and reference files
    pre_files = list(taxonomy_root.glob('linkbases/*/pre_*_role-*.xml'))
    ref_files = list(taxonomy_root.glob('linkbases/*/ref_*.xml'))
    
    print(f"Found {len(pre_files)} presentation files")
    print(f"Found {len(ref_files)} reference files")
    
    # Parse files
    print("Parsing XSD schema...")
    elements, roles = parse_xsd_file(xsd_file)
    print(f"  Found {len(elements)} elements")
    
    print("Parsing labels...")
    labels = parse_label_file(label_file) if label_file else {}
    print(f"  Found labels for {len(labels)} elements")
    
    print("Parsing documentation...")
    documentation = parse_documentation_file(doc_file) if doc_file else {}
    print(f"  Found documentation for {len(documentation)} elements")
    
    print("Parsing references...")
    all_references = {}
    for ref_file in ref_files:
        refs = parse_reference_file(ref_file)
        all_references.update(refs)
    print(f"  Found references for {len(all_references)} elements")
    
    print("Parsing presentation hierarchy...")
    all_relationships = []
    for pre_file in pre_files:
        relationships = parse_presentation_file(pre_file)
        all_relationships.extend(relationships)
    print(f"  Found {len(all_relationships)} parent-child relationships")
    
    children_map, parent_map, roots, all_hierarchy_elements = build_hierarchy(all_relationships)
    print(f"  Built hierarchy with {len(roots)} root elements")
    
    # Build JSON output
    print("\nBuilding JSON structure...")
    taxonomy_json = {}
    element_counter = 1
    processed = 0
    
    for element_name, element_info in elements.items():
        # Skip abstract elements
        if element_info.get('abstract', False):
            continue
        
        # Get primary label and variants
        element_labels = labels.get(element_name, [])
        pref_label = None
        label_variants = []
        
        for lbl in element_labels:
            role = lbl.get('role', '')
            text = lbl.get('text', '').strip()
            if not text:
                continue
            
            if '/role/label' in role and 'verbose' not in role and 'terse' not in role and 'documentation' not in role:
                if pref_label is None:
                    pref_label = text
                elif text != pref_label:
                    label_variants.append(text)
        
        # If no label found, create one from element name
        if not pref_label:
            pref_label = element_name.replace('_', ' ').replace('-', ' ').title()
        
        # Get documentation/definition
        definitions = []
        if element_name in documentation:
            doc_text = documentation[element_name]
            if doc_text:
                definitions.append({
                    'text': doc_text,
                    'reference': ''
                })
        
        # Add references as additional definitions
        if element_name in all_references:
            for ref in all_references[element_name]:
                ref_text = ' '.join([f"{k}: {v}" for k, v in ref.items() if v])
                if ref_text and not any(d['text'] == ref_text for d in definitions):
                    definitions.append({
                        'text': ref_text,
                        'reference': ''
                    })
        
        # Get hierarchical path
        path_names, path_labels = get_path_to_root(element_name, parent_map, labels)
        
        # Generate UUIDs for path
        path_id = [str(uuid.uuid4()) for _ in path_labels]
        
        # Add root taxonomy identifier
        if path_labels:
            path_labels.append('IFRS SDS')
            path_id.append(str(uuid.uuid4()))
        
        # Create entry
        entry_id = str(uuid.uuid4())
        taxonomy_json[entry_id] = {
            'id': element_counter,
            'prefLabel': pref_label,
            'isLeaf': is_leaf_element(element_name, children_map),
            'definitions': definitions if definitions else [],
            'label_variants': label_variants,
            'ignore_case': True,
            'path_id': path_id,
            'path_label': path_labels,
            'tags': ['IFRS SDS']
        }
        
        element_counter += 1
        processed += 1
    
    # Write JSON file
    print(f"\nWriting JSON to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(taxonomy_json, f, indent=2, ensure_ascii=False)
    
    print(f"\nConversion complete!")
    print(f"  Total elements in XSD: {len(elements)}")
    print(f"  Processed (non-abstract): {processed}")
    print(f"  Created taxonomy entries: {len(taxonomy_json)}")
    
    return taxonomy_json


# Example usage
if __name__ == '__main__':
    taxonomy_path = 'data/ifrs_sds/ifrs_sds'  # Path to taxonomy root folder
    output_file = 'data/ifrs_sds_taxonomy.json'
    
    result = convert_taxonomy_to_json(taxonomy_path, output_file)
    
    # Print sample entries
    if result:
        print("\n" + "="*60)
        print("Sample entries:")
        print("="*60)
        
        # Show first 3 entries
        for i, (key, value) in enumerate(list(result.items())[:3]):
            print(f"\nEntry {i+1}:")
            print(f"  Label: {value['prefLabel']}")
            print(f"  Is Leaf: {value['isLeaf']}")
            print(f"  Path: {' > '.join(value['path_label'])}")
            if value['definitions']:
                print(f"  Definition: {value['definitions'][0]['text'][:100]}...")
            print()
