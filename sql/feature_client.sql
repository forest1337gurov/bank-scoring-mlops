SELECT 
	ApplicationId,
	MainDebt,LoanSerialNumber,
	IsRegistrationAddressCoincides,dpd.MaxOverdueDays90 ,
    PositionType,
    EducationType,
    OccupationType,
    MaritalStatus,
    Sex,
    OS,
    BirthDate ,
    FieldOfActivity,
    WorkExperience,
    toYear(ApplicationDate) - toYear(BirthDate) AS age,
    IsRegistrationAddressCoincides,
    IncomeAmount,
    case when IsRegistrationAddressCoincides = 1 then TimezoneA else TimezoneAA end as Timezone,
    case when ActualIsPrivateHouse is not null then ActualIsPrivateHouse else IsPrivateHouse end as ActualIsPrivateHouse ,
    CardBrand,
    CardType,
    CardLevel,
    case when IsPC =1 then 'ПК'
        when IsTablet = 1 then 'Планшет'
        when IsMobile = 1 then 'Смартфон' end as gadget
FROM risk_ch_db.client_data_495credit cdc 
LEFT JOIN risk_ch_db.dpd_495credit dpd on cdc.ApplicationId = dpd.ApplicationId
 where cdc.ApplicationStatus in ('LoanIssued', 'LoanReturned')
  and ApplicationDate >= '{start_date}'
  and ApplicationDate < '{end_date}'