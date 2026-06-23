import snapped, json
from typing import Optional, Set, Literal, List
from dataclasses import dataclass, field


@dataclass
class UserRecord:

    @dataclass
    class Address:
        street: str
        city: str
        postcode: Optional[str] = None

    @dataclass
    class Tag:
        key: str
        value: str = field(default="unset")

    name: str
    address: Address
    score: int
    nickname: Optional[Set[str]] = None
    prior_addresses: List[Address] = field(default_factory=list)
    status: Literal["active", "inactive", "pending"] = field(default="active")
    tags: list[Tag] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


def main():
    print(f"Running with {snapped.__label__} @ {snapped.__version__}")
    source = snapped.snap(UserRecord)
    print(json.dumps(source.schema, indent=4))
    target = snapped.unsnap(json.dumps(source.to_dict()))

    assert source.schema == snapped.snap((target)).schema

    print(target)


if __name__ == "__main__":
    main()
