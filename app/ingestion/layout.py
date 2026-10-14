"""Reconstruct text flow from PDF rectangles rather than interleaving columns."""


def reading_order(blocks, page_width):
    if not blocks:
        return ""
    narrow = [b for b in blocks if b[2] - b[0] < page_width * 0.65]
    clusters = []
    for block in sorted(narrow, key=lambda b: b[0]):
        if clusters and abs(block[0] - clusters[-1][0]) < 20:
            clusters[-1][1] += len(block[4])
        else:
            clusters.append([block[0], len(block[4])])
    total = sum(weight for _, weight in clusters)
    anchors = [x for x, weight in clusters if weight >= total * 0.2]
    if len(anchors) < 2 or any(b - a < page_width * 0.18 for a, b in zip(anchors, anchors[1:])):
        anchors = [min(block[0] for block in blocks)]
    wide = [b for b in blocks if b[5] == "table" or
            (len(anchors) > 1 and sum(b[0] - 3 <= x <= b[2] - 3 for x in anchors) > 1)]
    ordinary = [b for b in blocks if b not in wide]
    paragraphs = []
    lower = float("-inf")
    for separator in sorted(wide, key=lambda b: b[1]) + [None]:
        upper = separator[1] if separator else float("inf")
        region = [b for b in ordinary if lower <= b[1] < upper]
        for anchor in anchors:
            column = sorted([b for b in region if min(anchors, key=lambda x: abs(b[0] - x)) == anchor], key=lambda b: b[1])
            current = []
            prior = None
            for block in column:
                height = max(6, min(block[3] - block[1], 18))
                gap = block[1] - prior[3] if prior else 0
                heading_change = prior and len(prior[4].split()) < 12 and not prior[4].rstrip().endswith((".", "!", "?")) and prior[3] - prior[1] > height * 1.1
                if current and (gap > height * 0.5 or heading_change or block[4].startswith(("- ", "* ", "ROW "))):
                    paragraphs.append("\n".join(current))
                    current = []
                current.append(block[4])
                prior = block
            if current:
                paragraphs.append("\n".join(current))
        if separator:
            paragraphs.append(separator[4])
            lower = separator[1]
    return "\n\n".join(paragraphs)
