function test_initial_approximation(T)

    @testset "initial approximation | $(T)" begin

        degs = [3, 5, 7, 9, 13]

        @testset "initial approximation | integrator | $(T)" begin
            for deg in degs

                #legexp = LegendreExponential{T,deg}()
                method = ExpAndGram{T,deg}()
                for ndiff = 1:deg
                    A, B = integrator2AB(T, ndiff)
                    Φgt, Ugt = integrator_exp_and_gram_chol(T, ndiff)
                    Φ, U = similar(A), similar(A)
                    _Φ, _U = FHG._exp_and_gram_chol_init!(Φ, U, A, B, method)
                    @test Φ ≈ _Φ
                    @test U ≈ _U

                    @test Φ ≈ Φgt
                    @test Ugt ≈ U
                end
            end
        end

    end

end
