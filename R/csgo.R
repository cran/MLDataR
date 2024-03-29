#' csgo
#' @docType data
#' @keywords CounterStrike Global Offensive eSports
#' @format A data frame with 1,133 rows and 17 variables:
#' \describe{
#'   \item{map}{Map on which the match was played}
#'   \item{day}{Day of the month}
#'   \item{month}{Month of the year}
#'   \item{year}{Year}
#'   \item{date}{Date of match DD/MM/YYYY}
#'   \item{wait_time_s}{Time waited to find match}
#'   \item{match_time_s}{Total match length in seconds}
#'   \item{team_a_rounds}{Number of rounds played as Team A}
#'   \item{team_b_rounds}{Number of rounds played as Team B}
#'   \item{ping}{Maximum ping in milliseconds;the signal that's sent from one computer to another on the same network}
#'   \item{kills}{Number of kills accumulated in match; max 5 per round}
#'   \item{assists}{Number of assists accumulated in a match,inflicting oppononent with more than 50 percent damage,who is then killed by another player accumulated in match max 5 per round}
#'   \item{deaths}{Number of times player died during match;max 1 per round}
#'   \item{mvps}{Most Valuable Player award}
#'   \item{hs_percent}{Percentage of kills that were a result from a shot to opponent's head}
#'   \item{points}{Number of points accumulated during match. Apoints are gained from kills, assists,bomb defuses & bomb plants. Points are lost for sucicide and friendly kills}
#'   \item{result}{The result of the match, Win, Loss, Draw}
#'}
#' @source Extracted by Asif Laldin \email{a.laldin@nhs.net}, March-2019

"csgo"
