from pydantic import BaseModel, Field


class CreditApplication(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Возраст клиента")

    annual_income: float = Field(
        ...,
        gt=0,
        description="Годовой доход"
    )

    employment_years: float = Field(
        ...,
        ge=0,
        description="Стаж работы"
    )

    loan_amount: float = Field(
        ...,
        gt=0,
        description="Сумма кредита"
    )

    loan_term_months: int = Field(
        ...,
        gt=0,
        description="Срок кредита (месяцы)"
    )

    credit_score: int = Field(
        ...,
        ge=300,
        le=850,
        description="Кредитный рейтинг"
    )

    existing_loans: int = Field(
        ...,
        ge=0,
        description="Количество действующих кредитов"
    )

    debt_to_income: float = Field(
        ...,
        ge=0,
        le=1,
        description="Отношение долгов к доходу"
    )

    owns_house: bool = Field(
        ...,
        description="Есть собственное жильё"
    )

    has_dependents: bool = Field(
        ...,
        description="Есть иждивенцы"
    )