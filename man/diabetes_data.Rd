% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/diabetes_data.R
\docType{data}
\name{diabetes_data}
\alias{diabetes_data}
\title{Diabetes datasets}
\format{
A data frame with 520 rows and 17 variables:
\describe{
  \item{Age}{age of the patient presenting with diabetes}
  \item{Gender}{gender of the patient with diabetes}
  \item{ExcessUrination}{if the patient has a history of excessive urination}
  \item{Polydipsia}{abnormal thurst, accompanied by the excessive intake of water or fluid}
  \item{WeightLossSudden}{Sudden weight loss that has recently occured}
  \item{Fatigue}{Fatigue or weakness}
  \item{Polyphagia}{excessive or extreme hunger}
  \item{GenitalThrush}{patient has thrush fungus on or near their genital region}
  \item{BlurredVision}{history of blurred vision}
  \item{Itching}{skin itching}
  \item{Irritability}{general irritability and mood issues}
  \item{DelayHealing}{delayed healing of wounds}
  \item{PartialPsoriasis}{partial psoriasis on the body}
  \item{MuscleStiffness}{stiffness of the muscles}
  \item{Alopecia}{scalp alopecia and hair shedding}
  \item{Obesity}{Classified as obese}
  \item{DiabeticClass}{Class label to indicate whether the patient is diabetic or not}
}
}
\source{
Collected by Gary Hutson \email{hutsons-hacks@outlook.com}, Dec-2021
}
\usage{
diabetes_data
}
\description{
Diabetes datasets
}
\examples{
library(dplyr)
data(diabetes_data)
# Convert diabetes data to factor'
diabetes_data <- diabetes_data \%>\%
 glimpse() \%>\%
 mutate(DiabeticClass = as.factor(DiabeticClass))
 is.factor(diabetes_data$DiabeticClass)
}
\keyword{classification}
\keyword{diabetes}
\keyword{learning}
\keyword{machine}
