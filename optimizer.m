function results = optimizer(optim_func, optim_solver, approximator, op_options)
if isa(optim_func, 'function_handle')
    warning(['You are using *' optim_solver '* solver format to save optimization results. Please make sure that format is correct for your *@' char(optim_func) '* solver.'])
end
%% defining options and validations
if nargin < 4
    op_options = false;
end
if ~isfield(op_options, 'iterations')
    iterations = 1;
else
    iterations = op_options.iterations;
    if ~isa(iterations, 'double')
        error('*iterations* must be a "double" array');
    end
end
if ~isfield(op_options, 'approximator_input')
    approximator_check = false;
else
    approximator_check = true;
    approximator_input = op_options.approximator_input;
    if ~isa(approximator_input, 'cell') && size(approximator_input, 2) ~= 2
        error('*approximator_input* must be a 1x2 "cell" array');
    end
end
if ~isa(approximator, 'function_handle')
    approximator_check = approximator;
end
if ~isfield(op_options, 'abserr_compare')
    absolute_err = false;
else
    absolute_err = true;
    abserr_compare = op_options.abserr_compare;
    if ~isa(abserr_compare, 'double')
        error('*abserr_compare* must be a "double" array');
    elseif length(abserr_compare) ~= size(approximator_input{2},2)
        error(['Length of *abserr_compare* and 2nd arg of *approximator_input* must be equal.*approximator_input* length is ' int2str(size(approximator_input{2},2))])
    end
end
if ~isfield(op_options, 'file_update')
    check_update = false;
else
    check_update = true;
    file_update = op_options.file_update;
end
if ~isfield(op_options, 'hybridized')
    hybridized = false;
    hybridization = false;
else
    hybridization = true;
    hybridized = op_options.hybridized;
end
if ~isfield(op_options, 'accuracy_criteria')
    accuracy_criteria = false;
    accuracy_check = false;
    accuracy_checked = true;
else
    accuracy_criteria = op_options.accuracy_criteria;
    accuracy_check = true;
    accuracy_checked = false;
    if ~isa(accuracy_criteria, 'struct')
        error('*accuracy_criteria* must be a "struct" with valid fieldnames and values');
    end
end
if ~isfield(op_options, 'replace')
    replace = false;
else
    replace = op_options.replace;
    if ~isa(replace, 'logical')
        error('*replace* must be a "bolean/logical".');
    end
end
if ~isfield(op_options, 'save')
    file_save = false;
else
    file_save = true;
    file_name = op_options.save;
    if ~check_update && exist(file_name, 'file')
        error(['File with filename "' file_name '" already exist. Please add *file_update* to update file or save in another filename.'])
    end    
end
if isa(optim_func, 'function_handle')
    optimize = true;
elseif isa(optim_func, 'logical')
    optimize = false;
else
    error('*optimize* must be a "logical" of "function_handle".');
end
if ~isfield(op_options, 'residual_error')
    residual_error_check = false;
else
    residual_error_check = true;
    residual_error = op_options.residual_error;
    if ~isa(residual_error, 'function_handle')
        error('*residual_error* must be a "function_handle".');
    end
end

%% Running Optimization and Generating Tables
optim_result = struct;
if check_update
    load (file_update)
    results = fieldnames(load (file_update));
    results = eval(results{1});
else
    results = struct;
end
if hybridization
    load (hybridized)
    hybridized = fieldnames(load (hybridized));
    hybridized = eval(hybridized{1});
    if length(iterations) > size(hybridized.tables.x_tab, 1)
        error(['Size of *iterations* must be less than or equal to ' int2str(size(hybridized.tables.x_tab, 1)) '.'])
    end
end
if optimize
    for i=iterations
        if hybridization
            iter = i;
        elseif replace && check_update
            iter = i;
        elseif ~replace && ~check_update && ~isfield(results, 'optim_results')
            iter = 1;
        else
            try
                iter = size(fieldnames(results.optim_results), 1) + 1;
            catch
                error('*replace* option will not work without *file_update*')
            end
        end
        if hybridization
            hyrbidizer = hybridized.optim_results.(['optim_' num2str(iter)]).x;
        else
            hyrbidizer = false;
        end

        disp([num2str(i) ' of ' num2str(iterations(length(iterations))) '  Optimization is running...']);
        switch optim_solver
            case {'fminbnd', 'fminsearch', 'intlinprog', 'fzero', 'patternsearch', 'simulannealbnd'}
                [optim_result.x, optim_result.fval, optim_result.exitflag, optim_result.output] = optim_func(hyrbidizer);
            case 'fminunc'
                [optim_result.x, optim_result.fval, optim_result.exitflag, optim_result.output, optim_result.grad, optim_result.hessian] = optim_func(hyrbidizer);
            case {'linprog', 'quadprog', 'fseminf'}
                [optim_result.x, optim_result.fval, optim_result.exitflag, optim_result.output, optim_result.lambda] = optim_func(hyrbidizer);
            case 'fgoalattain'
                [optim_result.x, optim_result.fval, optim_result.attainfactor, optim_result.exitflag, optim_result.output, optim_result.lambda] = optim_func(hyrbidizer);
            case 'fmincon'
                [optim_result.x, optim_result.fval, optim_result.exitflag, optim_result.output, optim_result.lambda, optim_result.grad, optim_result.hessian] = optim_func(hyrbidizer);
            case 'fminimax'
                [optim_result.x, optim_result.fval, optim_result.maxfval, optim_result.exitflag, optim_result.output, optim_result.lambda] = optim_func(hyrbidizer);
            case 'fsolve'
                 [optim_result.x, optim_result.fval, optim_result.exitflag, optim_result.output, optim_result.jacobian]= optim_func(hyrbidizer);
            case {'lsqnonneg', 'lsqlin'}
                 [optim_result.x, optim_result.resnorm, optim_result.residual, optim_result.exitflag, optim_result.output, optim_result.lambda] = optim_func(hyrbidizer);
            case {'lsqnonlin', 'lsqcurvefit'}
                 [optim_result.x, optim_result.resnorm, optim_result.residual, optim_result.exitflag, optim_result.output, optim_result.lambda, optim_result.jacobian] = optim_func(hyrbidizer);
            case {'ga', 'gamultiobj'}
                 [optim_result.x, optim_result.fval, optim_result.exitflag, optim_result.output, optim_result.population, optim_result.score] = optim_func(hyrbidizer);
        end
        if accuracy_check
            ac_ch_fields = fieldnames(accuracy_criteria);
            for j = 1:length(ac_ch_fields)
                if optim_result.(ac_ch_fields{j}) > accuracy_criteria.(ac_ch_fields{j})
                    accuracy_checked = false;
                    disp(['Optimization "' int2str(iter) '" not reach to required accuracy criteria.'])
                    break
                else
                    accuracy_checked = true;
                end
            end
        end
        if accuracy_checked
            results.optim_results.(['optim_' num2str(iter)]) = optim_result;
            results.tables.x_tab(iter,:) = optim_result.x;
            if isfield(optim_result, 'fval')
                results.tables.fval_tab(iter,:) = optim_result.fval;
            end
            if approximator_check
                for j = 1:length(approximator_input{1})
                    results.tables.approximated(iter, :, j) = approximator(approximator_input{1}(j), approximator_input{2}, optim_result.x);
                    if residual_error_check
                        results.tables.residual_error(iter, :, j) = residual_error(approximator_input{1}(j), approximator_input{2}, optim_result.x);
                    end
                end
            end
            if ~approximator_check && residual_error_check
                for j = 1:length(approximator_input{1})
                    results.tables.residual_error(iter, :, j) = residual_error(approximator_input{1}(j), approximator_input{2}, optim_result.x);
                end
            end
        end
        if file_save
            if ~isfield(results, 'optim_results') && i == iterations(length(iterations))
                error("Optimizations doesn't meet with required *accuracy_criteria*. Please change *accuracy_criteria* or solver. Some times accuracy also matched with in random runs.")
            elseif isfield(results, 'optim_results')
                if hybridization
                    hybridresults = results;
                    save(file_name, 'hybridresults');
                else
                    optimresults = results;
                    save(file_name, 'optimresults');
                end
            end
        end
    end
end
%% Finding Best Weights and Absolute Error
if ~optimize && ~check_update
    if absolute_err
        error('*abserr_compare* will not work without *file_update* or *optimize*. Please add *file_update* or set *optimize* to be true.')
    else
        error('Please set any one of *optimze* and *abserr_compare*.')
    end
end
if ~optimize && check_update
    if approximator_check
        results.tables.approximated = zeros(size(optim_result, 1), length(approximator_input{2}), length(approximator_input{1}));
        if residual_error_check
            results.tables.residual_error = zeros(size(optim_result, 1), length(approximator_input{2}), length(approximator_input{1}));
        end
        optim_result = results.tables.x_tab;
        for i = 1:size(optim_result, 1)
            for j = 1:length(approximator_input{1})
                    results.tables.approximated(i, :, j) = approximator(approximator_input{1}(j), approximator_input{2}, optim_result(i,:));
                if residual_error_check
                    results.tables.residual_error(i, :, j) = residual_error(approximator_input{1}(j), approximator_input{2}, optim_result(i,:));
                end
            end
        end
    end
    if ~approximator_check && residual_error_check
        results.tables.residual_error = zeros(size(optim_result, 1), length(approximator_input{2}), length(approximator_input{1}));
        optim_result = results.tables.x_tab;
        for i = 1:size(optim_result, 1)
            for j = 1:length(approximator_input{1})
                results.tables.residual_error(i, :, j) = residual_error(approximator_input{1}(j), approximator_input{2}, optim_result(i,:));
            end
        end
    end
    approximator_check = true;
end
try
    if absolute_err && approximator_check && isfield(results, 'tables')
        abs_err = zeros(size(results.tables.approximated, 1), size(results.tables.approximated, 2), size(results.tables.approximated, 3));
        for i = 1:size(results.tables.approximated, 3)
            for j = 1:size(results.tables.approximated, 1)
                abs_err(j,:,i) = abs(results.tables.approximated(j,:,i) - abserr_compare);
            end
        end

        mean_abserr_bycol = mean(abs_err, 1);
        mean_abserr_byrow = mean(abs_err, 2);
        [~, aembc_loc] = min(mean_abserr_bycol);
        [~, aembr_loc] = min(mean_abserr_byrow);
        ki_sq_bycol=(mean_abserr_bycol.^2).^2.^2;
        ki_sq_byrow=(mean_abserr_byrow.^2).^2.^2;
        for i = 1:size(results.tables.approximated, 3)
            best_abserr_bycol(:,i) = abs_err(:, aembc_loc(i), i);
        end
        for i = 1:size(results.tables.approximated, 3)
            best_abserr_byrow(i,:) = abs_err(aembr_loc(i), :, i);
        end
        
        results.best_weights = results.tables.x_tab(aembr_loc(1), :);

        results.abs_errors.abs_err = abs_err;
        results.abs_errors.mean_abserr_bycol = mean_abserr_bycol;
        results.abs_errors.mean_abserr_byrow = mean_abserr_byrow;
        results.abs_errors.best_abserr_bycol = best_abserr_bycol;
        results.abs_errors.best_abserr_byrow = best_abserr_byrow;
        results.abs_errors.ki_sq_bycol = ki_sq_bycol;
        results.abs_errors.ki_sq_byrow = ki_sq_byrow;
    end
catch ME
    warning(['warning in absoluter errors table calculating: ' ME.message '. Absolute Errors will not be calculated.'])
end
if ~isfield(results, 'optim_results')
   results = "Optimizations doesn't meet with required *accuracy_criteria*. Please change *accuracy_criteria* or solver. Some times accuracy also matched with in random runs.";
   error(results);
else
    if file_save
        if hybridization
            hybridresults = results;
            save(file_name, 'hybridresults');
        else
            optimresults = results;
            save(file_name, 'optimresults');
        end
        disp(['Result has saved in file "' file_name '"'])
    end
    disp('Optimization has completed.');
end
