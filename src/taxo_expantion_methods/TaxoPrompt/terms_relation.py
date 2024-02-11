from enum import Enum


class Relation(Enum):
    PARENT_OF = 'parent-of'
    CHILD_OF = 'child-of'
    SIBLING_OF = 'sibling-of'
    NEPHEW_OF = 'nephew-of'
