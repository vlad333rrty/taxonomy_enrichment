from enum import Enum


class Relation(Enum):
    PARENT_OF = 'parentOf'
    CHILD_OF = 'childOf'
    SIBLING_OF = 'siblingOf'
    NEPHEW_OF = 'nephewOf'
