import numpy as np
from utils import eval_auc_queries, eval_perc_queries, eval_auc_queries_spa_sem_lift, eval_perc_queries_spa_sem_lift
import torch

def check_conv(vals, window=2, tol=1e-6):
    '''
    Check the convergence of mode based on the evaluation score:
    Args:
        vals: a list of evaluation score
        window: the average window size
        tol: the threshold for convergence
    '''
    if len(vals) < 2 * window:
        return False
    conv = np.mean(vals[-window:]) - np.mean(vals[-2*window:-window]) 
    return conv < tol

def update_loss(loss, losses, ema_loss, ema_alpha=0.01):
    losses.append(loss)
    if ema_loss is None:
        ema_loss = loss
    else:
        ema_loss = (1-ema_alpha)*ema_loss + ema_alpha*loss
    return losses, ema_loss

def run_eval(model, queries, iteration, logger, by_type=False, geo_train = False, eval_detail_log = False):
    '''
    Given queries, evaluate AUC and APR by negative sampling and hard negative sampling
    Args:
        queries: 
            key: "full_neg" (full negative sample) or "one_neg" (only one negative sample)
            value: a dict()
                key: query type
                value: a dict()
                    key: formula template
                    value: the query object
        eval_detail_log: 
            whether to return the detail AUC eval result for each formula in each query type
    Return:
        vals: a dict()
            key: query type, or query_type+"hard"
            value: AUC for this query type
        qtype2fm_auc: a dict()
            key: query type
            valie: a dict()
                key: (formula.query_type, formula.rels)
                value: AUC for this formula
    '''
    vals = {}
    aprs = {}

    qtype2fm_auc = {}
    qtype2fm_q_prec = {}
    auc_list = []
    if geo_train:
        geo_msg = "GEO"
    else:
        geo_msg = ""
    def _print_by_rel(rel_aucs, logger):
        for rels, auc in rel_aucs.iteritems():
            logger.info(str(rels) + "\t" + str(auc))
    for query_type in queries["one_neg"]:
        # for each query type, do normal negative sampling
        # reauc_aucs: a dict():
        #     key: fomula
        #     value: AUC for this formula
        # auc: overall AUC score for all test queries for current query type
        auc, reauc_aucs = eval_auc_queries(queries["one_neg"][query_type], model)
        # perc: average percentiel rank (APR) for current query type
        perc_tuple = eval_perc_queries(queries["full_neg"][query_type], model, 
                                    eval_detail_log = eval_detail_log)
        if eval_detail_log:
            # perc: average percentiel rank (APR) for current query type
            '''
            fm2query_prec: a dict()
                key: (formula.query_type, formula.rels)
                value: a list, [query.serialize(), prec]
                    query.serialize(): (query_graph, neg_samples, hard_neg_samples)
                    prec: prec score for current query
            '''
            perc, fm2query_prec = perc_tuple
            qtype2fm_q_prec[query_type] = fm2query_prec
            qtype2fm_auc[query_type] = reauc_aucs
        else:
            # perc: average percentiel rank (APR) for current query type
            perc = perc_tuple
        vals[query_type] = auc
        aprs[query_type] = perc
        
        logger.info("Eval {:s}: {:s} val AUC: {:f} val APR {:f}; iteration: {:d}".format(geo_msg, query_type, auc, perc, iteration))
        if by_type:
            _print_by_rel(rel_aucs, logger)
        if "inter" in query_type:
            # for "inter" query type, do hard negative sampling
            auc, rel_aucs = eval_auc_queries(queries["one_neg"][query_type], model, hard_negatives=True)
            perc_tuple = eval_perc_queries(queries["full_neg"][query_type], model, hard_negatives=True,
                                    eval_detail_log = eval_detail_log)
            if eval_detail_log:
                # perc: average percentiel rank (APR) for current query type
                '''
                fm2query_prec: a dict()
                    key: (formula.query_type, formula.rels)
                    value: a list, [query.serialize(), prec]
                        query.serialize(): (query_graph, neg_samples, hard_neg_samples)
                        prec: prec score for current query
                '''
                perc, fm2query_prec = perc_tuple
                qtype2fm_q_prec[query_type + ":hard"] = fm2query_prec
                qtype2fm_auc[query_type + ":hard"] = rel_aucs
            else:
                # perc: average percentiel rank (APR) for current query type
                perc = perc_tuple

            logger.info("Eval {:s}: Hard-{:s} val AUC: {:f} val APR {:f}; iteration: {:d}".format(geo_msg, query_type, auc, perc, iteration))
            if by_type:
                _print_by_rel(rel_aucs, logger)
            vals[query_type + "hard"] = auc
            aprs[query_type + "hard"] = perc
            
    if eval_detail_log:
        return vals, aprs, qtype2fm_auc, qtype2fm_q_prec
    else:
        return vals, aprs

def run_train(model, optimizer, 
        train_queries, val_queries, test_queries, 
        logger = None,
        max_burn_in =100000, batch_size=512, log_every=100, val_every=1000, tol=1e-6,
        max_iter=int(10e7), inter_weight=0.005, path_weight=0.01, model_file=None, edge_conv=False, geo_train = False, val_queries_geo = None, test_queries_geo = None):
    '''
    Args:
        train_queries:
            key: query type
            value: a dict()
                key: formula template
                value: the query object
        val_queries/test_queries/val_queries_geo/test_queries_geo:
            # val_queries_geo and test_queries_geo DO NOT have 1-chain query
            key: "full_neg" (full negative sample) or "one_neg" (only one negative sample)
            value: a dict()
                key: query type
                value: a dict()
                    key: (formula.query_type, formula.rels)
                    value: the query object
        geo_train: whether we train/val/test using geographic queries
    '''
    if geo_train:
        assert val_queries_geo is not None
        assert test_queries_geo is not None
    ema_loss = None
    vals = []
    losses = []
    conv_test = None
    conv_test_apr = None
    conv_geo_test = None
    conv_geo_test_apr = None
    for i in xrange(max_iter):
        
        optimizer.zero_grad()
        loss = run_batch(train_queries["1-chain"], model, i, batch_size)
        # if edge_conv=False and (the model is edge converge or len(losses) >= max_burn_in)
        if not edge_conv and (check_conv(vals, tol=tol) or len(losses) >= max_burn_in):
            logger.info("Edge converged at iteration {:d}".format(i-1))
            logger.info("Testing at edge conv...")
            conv_test, conv_test_apr = run_eval(model, test_queries, i, logger)
            conv_test = np.mean(conv_test.values())
            conv_test_apr = np.mean(conv_test_apr.values())

            if geo_train:
                logger.info("geo query...")
                conv_geo_test, conv_geo_test_apr = run_eval(model, test_queries_geo, i, logger, geo_train = True)
                conv_geo_test = np.mean(conv_geo_test.values())
                conv_geo_test_apr = np.mean(conv_geo_test_apr.values())

            edge_conv = True
            losses = []
            ema_loss = None
            vals = []
            if not model_file is None:
                torch.save(model.state_dict(), model_file.replace(".pth", "--edge_conv.pth"))
        
        if edge_conv:
            for query_type in train_queries:
                if query_type == "1-chain":
                    continue
                if "inter" in query_type:
                    loss += inter_weight*run_batch(train_queries[query_type], model, i, batch_size)
                    loss += inter_weight*run_batch(train_queries[query_type], model, i, batch_size, hard_negatives=True)
                else:
                    loss += path_weight*run_batch(train_queries[query_type], model, i, batch_size)
            if check_conv(vals, tol=tol):
                logger.info("Fully converged at iteration {:d}".format(i))
                break
        # print(loss)
        # print(loss.item())
        # losses, ema_loss = update_loss(loss.data[0], losses, ema_loss)
        losses, ema_loss = update_loss(loss.item(), losses, ema_loss)
        loss.backward()
        optimizer.step()
            
        if i % log_every == 0:
            logger.info("Iter: {:d}; ema_loss: {:f}".format(i, ema_loss))
            
        if i >= val_every and i % val_every == 0:
            v, aprs = run_eval(model, val_queries, i, logger)
            logger.info("Validate macro-averaged AUC: {:f}, APR: {:f}".format(np.mean(v.values()), np.mean(aprs.values())))
            if geo_train:
                logger.info("geo query...")
                v_geo, aprs_geo = run_eval(model, val_queries_geo, i, logger, geo_train = True)
                logger.info("GEO Validate macro-averaged AUC: {:f}, APR: {:f}".format(np.mean(v_geo.values()), np.mean(aprs_geo.values())))

            if edge_conv:
                if geo_train:
                    # vals.append(np.mean(v.values() + v_geo.values()))
                    vals.append(np.mean(v_geo.values()))
                else:
                    vals.append(np.mean(v.values()))
                if not model_file is None:
                    torch.save(model.state_dict(), model_file)
            else:
                # val_queries_geo and test_queries_geo DO NOT have 1-chain query
                vals.append(v["1-chain"])
                if not model_file is None:
                    torch.save(model.state_dict(), model_file.replace(".pth", "--edge_conv.pth"))
    
    v, aprs = run_eval(model, test_queries, i, logger)
    logger.info("Test macro-averaged AUC: {:f}, APR: {:f}".format(np.mean(v.values()), np.mean(aprs.values())))
    if conv_test is not None:
        logger.info("AUC Improvement from edge conv: {:f}".format((np.mean(v.values())-conv_test)/conv_test))
        logger.info("APR Improvement from edge conv: {:f}".format((np.mean(aprs.values())-conv_test_apr)/conv_test_apr))

    if geo_train:
        logger.info("geo query...")
        v_geo, aprs_geo = run_eval(model, test_queries_geo, i, logger, geo_train = True)
        logger.info("GEO Test macro-averaged AUC: {:f}, APR: {:f}".format(np.mean(v_geo.values()), np.mean(aprs_geo.values())))
        if conv_geo_test is not None:
            logger.info("GEO AUC Improvement from edge conv: {:f}".format((np.mean(v_geo.values())-conv_geo_test)/conv_geo_test))
            logger.info("GEO APR Improvement from edge conv: {:f}".format((np.mean(aprs_geo.values())-conv_geo_test_apr)/conv_geo_test_apr))

def run_batch(train_queries, enc_dec, iter_count, batch_size, hard_negatives=False):
    '''
    Given the training queries and the iterator num, find the query batch and train encoder-decoder
    Args:
        train_queries: a dict()
            key: formula template
            value: the query object
        enc_dec: encoder-decoder model
        iter_count: scaler, iterator num
        batch_size: 
        hard_negatives: True/False
    '''
    # num_queries: a list of num of queries per formula
    num_queries = [float(len(queries)) for queries in train_queries.values()]
    denom = float(sum(num_queries))
    # Use the num of queries per formula to form a multinomial dist to randomly pick on value
    formula_index = np.argmax(np.random.multinomial(1, 
            np.array(num_queries)/denom))
    formula = train_queries.keys()[formula_index]
    n = len(train_queries[formula])
    start = (iter_count * batch_size) % n
    end = min(((iter_count+1) * batch_size) % n, n)
    end = n if end <= start else end
    # print("start: {:d}\tend: {:d}".format(start, end))
    queries = train_queries[formula][start:end]
    loss = enc_dec.margin_loss(formula, queries, hard_negatives=hard_negatives)
    return loss


def run_eval_spa_sem_lift(model, queries, iteration, logger, by_type=False, 
            geo_train = False, eval_detail_log = False, do_spa_sem_lift = False):
    '''
    Given queries, evaluate AUC and APR by negative sampling and hard negative sampling
    Args:
        queries: 
            key: "full_neg" (full negative sample) or "one_neg" (only one negative sample)
            value: a dict()
                key: query type
                value: a dict()
                    key: formula template
                    value: the query object
        eval_detail_log: 
            whether to return the detail AUC eval result for each formula in each query type
    Return:
        vals: a dict()
            key: query type, or query_type+"hard"
            value: AUC for this query type
        qtype2fm_auc: a dict()
            key: query type
            valie: a dict()
                key: (formula.query_type, formula.rels)
                value: AUC for this formula
    '''
    vals = {}
    aprs = {}

    qtype2fm_auc = {}
    qtype2fm_q_prec = {}
    auc_list = []
    if geo_train:
        geo_msg = "GEO"
    else:
        geo_msg = ""
    def _print_by_rel(rel_aucs, logger):
        for rels, auc in rel_aucs.iteritems():
            logger.info(str(rels) + "\t" + str(auc))
    for query_type in queries["one_neg"]:
        # for each query type, do normal negative sampling
        # reauc_aucs: a dict():
        #     key: fomula
        #     value: AUC for this formula
        # auc: overall AUC score for all test queries for current query type
        auc, reauc_aucs = eval_auc_queries_spa_sem_lift(queries["one_neg"][query_type], model, do_spa_sem_lift = do_spa_sem_lift)
        # perc: average percentiel rank (APR) for current query type
        perc_tuple = eval_perc_queries_spa_sem_lift(queries["full_neg"][query_type], model, 
                                    eval_detail_log = eval_detail_log, do_spa_sem_lift = do_spa_sem_lift)
        if eval_detail_log:
            # perc: average percentiel rank (APR) for current query type
            '''
            fm2query_prec: a dict()
                key: (formula.query_type, formula.rels)
                value: a list, [query.serialize(), prec]
                    query.serialize(): (query_graph, neg_samples, hard_neg_samples)
                    prec: prec score for current query
            '''
            perc, fm2query_prec = perc_tuple
            qtype2fm_q_prec[query_type] = fm2query_prec
            qtype2fm_auc[query_type] = reauc_aucs
        else:
            # perc: average percentiel rank (APR) for current query type
            perc = perc_tuple
        vals[query_type] = auc
        aprs[query_type] = perc
        
        logger.info("Eval {:s}: {:s} val AUC: {:f} val APR {:f}; iteration: {:d}".format(geo_msg, query_type, auc, perc, iteration))
        if by_type:
            _print_by_rel(rel_aucs, logger)
            
    if eval_detail_log:
        return vals, aprs, qtype2fm_auc, qtype2fm_q_prec
    else:
        return vals, aprs

def run_batch_spa_sem_lift(train_queries, enc_dec, iter_count, batch_size, 
                hard_negatives=False, do_spa_sem_lift = False):
    '''
    Given the training queries and the iterator num, find the query batch and train encoder-decoder
    Args:
        train_queries: a dict()
            key: formula template
            value: the query object
        enc_dec: encoder-decoder model
        iter_count: scaler, iterator num
        batch_size: 
        hard_negatives: True/False
    '''
    # num_queries: a list of num of queries per formula
    num_queries = [float(len(queries)) for queries in train_queries.values()]
    denom = float(sum(num_queries))
    # Use the num of queries per formula to form a multinomial dist to randomly pick on value
    formula_index = np.argmax(np.random.multinomial(1, 
            np.array(num_queries)/denom))
    formula = train_queries.keys()[formula_index]
    n = len(train_queries[formula])
    start = (iter_count * batch_size) % n
    end = min(((iter_count+1) * batch_size) % n, n)
    end = n if end <= start else end
    # print("start: {:d}\tend: {:d}".format(start, end))
    queries = train_queries[formula][start:end]
    loss = enc_dec.margin_loss(formula, queries, hard_negatives=hard_negatives, do_spa_sem_lift = do_spa_sem_lift)
    return loss


def run_train_spa_sem_lift(model, optimizer, 
        train_queries, val_queries, test_queries, 
        logger = None,
        max_burn_in =100000, batch_size=512, log_every=100, val_every=1000, tol=1e-6,
        max_iter=int(10e7), inter_weight=0.005, path_weight=0.01, model_file=None, edge_conv=False, 
        geo_train = False, 
        spa_sem_lift_loss_weight = 1.0,
        train_queries_geo = None, val_queries_geo = None, test_queries_geo = None):
    '''
    Args:
        train_queries:
            key: query type
            value: a dict()
                key: formula template
                value: the query object
        val_queries/test_queries/val_queries_geo/test_queries_geo:
            # val_queries_geo and test_queries_geo DO NOT have 1-chain query
            key: "full_neg" (full negative sample) or "one_neg" (only one negative sample)
            value: a dict()
                key: query type
                value: a dict()
                    key: (formula.query_type, formula.rels)
                    value: the query object
        geo_train: whether we train/val/test using geographic queries
    '''
    
    assert train_queries_geo is not None
    assert val_queries_geo is not None
    assert test_queries_geo is not None
    ema_loss = None
    vals = []
    losses = []
    
    
    for i in xrange(max_iter):
        
        optimizer.zero_grad()
        # we train the normal link prediction objective
        loss = run_batch_spa_sem_lift(train_queries["1-chain"], model, i, batch_size, do_spa_sem_lift = False)

        # train the semantic lifting objective
        loss = spa_sem_lift_loss_weight*run_batch_spa_sem_lift(train_queries_geo["1-chain"], model, i, batch_size, do_spa_sem_lift = True)
        
        if check_conv(vals, tol=tol):
            logger.info("Fully converged at iteration {:d}".format(i))
            break


        # print(loss)
        # print(loss.item())
        # losses, ema_loss = update_loss(loss.data[0], losses, ema_loss)
        losses, ema_loss = update_loss(loss.item(), losses, ema_loss)
        loss.backward()
        optimizer.step()
            
        if i % log_every == 0:
            logger.info("Iter: {:d}; ema_loss: {:f}".format(i, ema_loss))
            
        if i >= val_every and i % val_every == 0:
            # link prediction eval
            v, aprs = run_eval_spa_sem_lift(model, val_queries, i, logger, do_spa_sem_lift = False)
            logger.info("Validate macro-averaged AUC: {:f}, APR: {:f}".format(np.mean(v.values()), np.mean(aprs.values())))
            
            logger.info("geo query...")
            # spatial semantic lifting eval
            v_geo, aprs_geo = run_eval_spa_sem_lift(model, val_queries_geo, i, logger, geo_train = True, do_spa_sem_lift = True)
            logger.info("GEO Validate macro-averaged AUC: {:f}, APR: {:f}".format(np.mean(v_geo.values()), np.mean(aprs_geo.values())))

            
            if geo_train:
                vals.append(np.mean(v.values() + v_geo.values()))
                # vals.append(np.mean(v_geo.values()))
            else:
                vals.append(np.mean(v.values()))
            if not model_file is None:
                torch.save(model.state_dict(), model_file)
       
    
    v, aprs = run_eval_spa_sem_lift(model, test_queries, i, logger, do_spa_sem_lift = False)
    logger.info("Test macro-averaged AUC: {:f}, APR: {:f}".format(np.mean(v.values()), np.mean(aprs.values())))
    
    
    logger.info("geo query...")
    v_geo, aprs_geo = run_eval_spa_sem_lift(model, test_queries_geo, i, logger, geo_train = True, do_spa_sem_lift = True)
    logger.info("GEO Test macro-averaged AUC: {:f}, APR: {:f}".format(np.mean(v_geo.values()), np.mean(aprs_geo.values())))
    