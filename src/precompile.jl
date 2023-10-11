@setup_workload begin 

    function precompile_test_problem(T, n, m)
        A = randn(T, n, n)
        B = randn(T, n, m) 
        return A, B 
    end

    function precompile_method(method, T, n, m) 
        A, B = precompile_test_problem(T, n, m) 
        _, _ = exp_and_gram_chol(A, B, method) 
        _, _ = exp_and_gram(A, B, method)
    end

    Ts = [Float64]
    qs = [3, 5, 7, 9, 13]
    n, m = 2, 1 

    @compile_workload begin
        for T in Ts 
            for q in qs 
                method = ExpAndGram{T, q}()
                precompile_method(method, T, n, m) 
            end 
            method = AdaptiveExpAndGram{T}()
            precompile_method(method, T, n, m) 
        end
    end

end