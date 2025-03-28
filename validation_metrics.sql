/*

This function computes some metrics to assess the performance of a model.

PARAMS
------
  * table_source : name of the table with the scores and the target of the model. Two columns are mandatory :
    * score
    * target

It creates a table scoring.roc with some metrics like precision, recall, fall-out and it outputs the area under the ROC and Precision/Recall curve.

*/

CREATE OR REPLACE FUNCTION validation_metrics (table_source TEXT) RETURNS DOUBLE PRECISION[] AS $$
DECLARE
	metrics DOUBLE PRECISION[];
BEGIN
	EXECUTE '
		DROP TABLE IF EXISTS scoring.roc;
		CREATE TABLE scoring.roc WITH (appendonly=true, orientation=column) AS
			WITH cutoffs AS (
        		SELECT 
          			score AS cutoff,
          			SUM(target) AS pos,
          			COUNT(*) - SUM(target) AS neg
        		FROM ' || table_source || '
        		GROUP BY cutoff  
      		), metrics_tmp AS (
        		SELECT 
		          	cutoff, 
		          	SUM(pos) OVER () as n_pos,
		          	SUM(neg) OVER () as n_neg,
		          	SUM(pos) OVER (ORDER BY cutoff DESC) as tp,
		          	SUM(neg) OVER (ORDER BY cutoff DESC) as fp
					--,SUM(neg) OVER () - SUM(neg) OVER (ORDER BY cutoff DESC) as tn,
					--SUM(pos) OVER () - SUM(pos) OVER (ORDER BY cutoff DESC) as fn
        		FROM cutoffs
      		), metrics as (
        		SELECT *,
		        	tp / n_pos ::double precision AS tpr,
		          	fp / n_neg AS fpr,
		          	tp / (tp + fp) ::double precision AS prec
		          	--,(tp + fp) / (tp +fp + tn +fn) ::double precision AS rpp,
		          	--(tp / n_pos) / ((tp + fp) / (tp +fp + tn +fn)) ::double precision as lift
		        FROM metrics_tmp
      		)
			SELECT
        		*,
        		COALESCE(lag(tpr) OVER (ORDER BY cutoff DESC), 0) AS tpr_lag,
        		COALESCE(lag(fpr) OVER (ORDER BY cutoff DESC), 0) AS fpr_lag,
        		COALESCE(lag(prec) OVER (ORDER BY cutoff DESC), 0) AS prec_lag
				--,COALESCE(lag(rpp) OVER (ORDER BY cutoff DESC), 0) AS rpp_lag,
				--COALESCE(lag(lift) OVER (ORDER BY cutoff DESC), 0) AS lift_lag
      		FROM rates
      	DISTRIBUTED BY (cutoff);';

    EXECUTE '
    	SELECT 
  			ARRAY[
  				SUM((tpr + tpr_lag) * (fpr - fpr_lag) / 2), --auc
  				SUM((prec + prec_lag) * (tpr - tpr_lag) / 2) --auc (precision/recall)
  				--,SUM((tpr + tpr_lag) * (rpp - rpp_lag) / 2) --auc (lift)
  			]
		FROM scoring.roc;' 
	INTO metrics;

	RETURN metrics;
END;
$$ LANGUAGE plpgsql;