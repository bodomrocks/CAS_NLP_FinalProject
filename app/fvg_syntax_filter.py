"""
Syntaktische Nachfilterung für FVG-Vorhersagen (v2).

Nach der BERT-Inferenz prüft spaCy, ob zwischen dem vorhergesagten
B-VERB-Token und dem B-NOM-Token eine FVG-kompatible Dependenzrelation
besteht. Paare ohne syntaktischen Bezug werden verworfen.

Erkannte Relationen (TIGER-Schema, de_core_news_lg):
  - Direktes Objekt: NOM.dep_ in OBJ_DEPS, NOM.head == VERB
  - Präpositional:   NOM.dep_ == "nk", NOM.head.dep_ in PP_DEPS,
                     NOM.head.head == VERB

Neue Prüfungen (v2):
  - Klauselgrenze: NOM muss VERB erreichen, ohne eine Satzgrenze
                   (rc, oc, neb, om) zu überschreiten. Verhindert,
                   dass Verb und Nomen aus verschiedenen Teilsätzen
                   fälschlicherweise gepaart werden.
  - Präpositions-Whitelist: Nur FVG-typische Präpositionen gelten als
                   gültige Brücken zwischen NOM und VERB (nicht "ohne",
                   "trotz", "wegen" etc.).
"""

OBJ_DEPS = {"oa", "oa2", "obj", "dobj"}
PP_DEPS  = {"mo", "cvc", "op", "pg", "obl"}

# Überqueren dieser Dependenz-Relationen markiert eine Satzgrenze
CLAUSE_BOUNDARY_DEPS = {"rc", "oc", "neb", "om", "uc"}

# Präpositionen, die typischerweise Teil eines FVG sind.
# "ohne", "trotz", "wegen", "während" signalisieren freie Adjunkte, keine FVG.
FVG_PREPS = {
    "in", "zu", "zur", "zum", "an", "auf", "unter", "über",
    "von", "vor", "nach", "bei", "mit", "gegen", "für",
    "ausser",  # ausser Kraft setzen/treten, ausser Acht lassen
}


def _tokens_in_span(doc, start: int, end: int):
    """Gibt spaCy-Tokens zurück, die den Zeichenbereich [start, end) überlappen."""
    return [t for t in doc if not t.is_space
            and t.idx < end and t.idx + len(t.text) > start]


def _same_clause(verb_tok, nom_tok) -> bool:
    """
    True wenn nom_tok den verb_tok erreicht, ohne eine Satzgrenze zu
    überschreiten (rc, oc, neb, om). Verhindert Kreuzklausel-Paarungen
    wie Verb im Hauptsatz + Nomen im Relativsatz.
    """
    t = nom_tok
    visited: set[int] = set()
    while t.head != t and id(t) not in visited:
        visited.add(id(t))
        if t == verb_tok:
            return True
        if t.dep_ in CLAUSE_BOUNDARY_DEPS:
            return False
        t = t.head
    return t == verb_tok


def _is_fvg_dep(verb_tok, nom_tok) -> bool:
    """True wenn nom_tok eine FVG-Dependenz zu verb_tok hat."""
    if verb_tok is None or nom_tok is None:
        return False

    # Klauselgrenzen-Check: NOM und VERB müssen im selben Teilsatz liegen
    if not _same_clause(verb_tok, nom_tok):
        return False

    # Direktes Objekt
    if nom_tok.head == verb_tok and nom_tok.dep_ in OBJ_DEPS:
        return True

    # Via Präposition: NOM → Präp → VERB
    if (nom_tok.dep_ == "nk"
            and nom_tok.head.head == verb_tok
            and nom_tok.head.dep_ in PP_DEPS):
        # Präpositions-Whitelist: nur FVG-typische Präpositionen
        prep_lemma = nom_tok.head.lemma_.lower()
        if prep_lemma in FVG_PREPS:
            return True

    return False


def syntactic_filter(text: str, entities: list[dict], nlp) -> list[dict]:
    """
    Filtert BERT-Entitäten anhand des spaCy-Dependency-Graphs.

    Behält nur (VERB, NOM)-Paare, zwischen denen eine FVG-typische
    syntaktische Relation besteht. Entitäten ohne gültige Paarung
    werden verworfen.

    Parameters
    ----------
    text     : Originaltext (gleicher String wie für BERT-Inferenz)
    entities : Ausgabe der transformers-Pipeline (aggregation_strategy="first")
    nlp      : geladenes spaCy-Modell

    Returns
    -------
    Gefilterte Entitätsliste (möglicherweise leer).
    """
    if not entities:
        return entities

    doc = nlp(text)

    verb_ents = [e for e in entities if e["entity_group"] == "VERB"]
    nom_ents  = [e for e in entities if e["entity_group"] == "NOM"]

    if not verb_ents or not nom_ents:
        return []

    valid_verb_ids: set[int] = set()
    valid_nom_ids:  set[int] = set()

    for ve in verb_ents:
        v_cands = _tokens_in_span(doc, ve["start"], ve["end"])
        # Bevorzuge echte VERB-Token (Partizip, Infinitiv, Finitum)
        v_verbs = [t for t in v_cands if t.pos_ == "VERB"]
        v_tok = v_verbs[0] if v_verbs else (v_cands[0] if v_cands else None)

        for ne in nom_ents:
            n_cands = _tokens_in_span(doc, ne["start"], ne["end"])
            # Bevorzuge NOUN/PROPN innerhalb der NOM-Span
            n_nouns = [t for t in n_cands if t.pos_ in ("NOUN", "PROPN")]
            n_tok = n_nouns[0] if n_nouns else (n_cands[0] if n_cands else None)

            if _is_fvg_dep(v_tok, n_tok):
                valid_verb_ids.add(id(ve))
                valid_nom_ids.add(id(ne))

    return [e for e in entities
            if id(e) in valid_verb_ids or id(e) in valid_nom_ids]
