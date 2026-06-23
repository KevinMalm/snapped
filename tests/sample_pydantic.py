import snapped, json
from typing import Optional, Set, Literal, List, Annotated
from pydantic import BaseModel, Field

HexColor = Annotated[str, Field(pattern=r"^#[0-9A-Fa-f]{6}$")]


class UserRecord(BaseModel):
    """Profile details for an individual user"""

    class Profile(BaseModel):
        color: HexColor

    class Address(BaseModel):
        """An Address"""

        street: str
        city: str
        postcode: Optional[str] = None

    class Tag(BaseModel):
        key: str
        value: str = Field(
            default="unset", description="Tag value, defaults to 'unset'"
        )

    name: str = Field(alias="Name", description="The greeting target")
    nickname: Optional[Set[str]] = None
    status: Literal["active", "inactive", "pending"] = Field(
        default="active",
        description="Current status of the entity",
    )
    address: Address
    prior_addresses: List[Address] = Field(default_factory=list)
    tags: list[Tag] = Field(
        default_factory=list, description="Arbitrary key/value tags"
    )
    score: int = Field(default=0, ge=0, le=100, description="Score between 0 and 100")
    metadata: dict[str, str] = Field(default_factory=dict)


def main():
    print(f"Running with {snapped.__label__} @ {snapped.__version__}")
    source = snapped.snap(UserRecord)
    target = snapped.unsnap(json.dumps(source.to_dict()))

    print(snapped.unsnap(snapped.snap(UserRecord.Profile))(color="#121212"))
    assert source.schema == snapped.snap((target)).schema


if __name__ == "__main__":
    main()
