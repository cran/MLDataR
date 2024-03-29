#' Care Home Incidents
#' @description a NHS patient safety incidents dataset: \url{https://www.england.nhs.uk/patient-safety/report-patient-safety-incident/} dataset that has been synthetically generated against real data
#' @docType data
#' @keywords care home incidents supervised machine learning classification
#' @format A data frame with 1216 rows and 12 variables:
#' \describe{
#'   \item{CareHomeFail}{a binary indicator to specify whether a certain care home is failing}
#'   \item{WeightLoss}{aggregation of incidents indicating weight loss in patient}
#'   \item{Medication}{medication missed aggregaation}
#'   \item{Falls}{Recorded number of patient falls}
#'   \item{Choking}{Number of patient choking incidents}
#'   \item{UnexpectedDeaths}{unexpected deaths in the care home}
#'   \item{Bruising}{Number of bruising incidents in the care home}
#'   \item{Absconsion}{Absconding from the care home setting}
#'   \item{ResidentAbuseByResident}{Abuse conducted by one care home resident against another}
#'   \item{ResidentAbuseByStaff}{Incidents of resident abuse by staff}
#'   \item{ResidentAbuseOnStaff}{Incidents of residents abusing staff}
#'   \item{Wounds}{Unexplained wounds against staff}
#'   }

#' @source Collected by Gary Hutson \email{hutsons-hacks@outlook.com}, Jan-2022
#' @examples
#' library(dplyr)
#' data(care_home_incidents)
#' # Convert diabetes data to factor'
#' ch_incs <- care_home_incidents %>%
#'  mutate(CareHomeFail = as.factor(CareHomeFail))
#'  ch_incs %>% glimpse()
#'  # Check factor
#'  factor(ch_incs$CareHomeFail)
"care_home_incidents"
