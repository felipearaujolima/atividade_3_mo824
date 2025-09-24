package problems.qbf.solvers;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

import metaheuristics.tabusearch.AbstractTS;
import problems.qbf.QBF;
import problems.qbf.QBF_Inverse;
import solutions.Solution;

/**
 * Metaheuristic TS (Tabu Search) for MAX-SC-QBF.
 * Implements parameterized tabu tenure, search method (first/best improving),
 * and different tabu strategies.
 *
 * Author: adaptado para atividade
 */
public class TS_QBF extends AbstractTS<Integer> {

    private final Integer fake = new Integer(-1);

    private boolean bestImproving; // true = best improving, false = first improving
    private String tabuStrategy; // "default", "intensificationRestart", "diversificationRestart", "strategicOscillation"
    private int[] usageCount;
    private double diversificationFixationRate = 0.2;
    private double diversificationSelectionProbability = 0.5;

    /**
     * Constructor
     * @param tenure tabu tenure
     * @param iterations max iterations
     * @param filename problem instance filename
     * @param bestImproving true para best improving, false para first improving
     * @param tabuStrategy string indicando a estratégia tabu
     * @throws IOException
     */
    public TS_QBF(Integer tenure, Integer iterations, String filename, boolean bestImproving, String tabuStrategy) throws IOException {
        super(new QBF_Inverse(filename), tenure, iterations);
        this.bestImproving = bestImproving;
        this.tabuStrategy = tabuStrategy;
        this.usageCount = new int[ObjFunction.getDomainSize()];
    }

    /**
     * Diversification by Restart Constructor
     * @param tenure tabu tenure
     * @param iterations max iterations
     * @param filename problem instance filename
     * @param bestImproving true para best improving, false para first improving
     * @param tabuStrategy string indicando a estratégia tabu
     * @param diversificationFixationRate controla quantas variáveis serão fixadas (varia de 0 a 1)
     * @param diversificationSelectionProbability probabilidade de uma variável ser selecionada na fase aleatória
     * @throws IOException
     */
    public TS_QBF(Integer tenure, Integer iterations, String filename, boolean bestImproving, String tabuStrategy,
                  double diversificationFixationRate, double diversificationSelectionProbability) throws IOException {
        this(tenure, iterations, filename, bestImproving, tabuStrategy);
        this.diversificationFixationRate = diversificationFixationRate;
        this.diversificationSelectionProbability = diversificationSelectionProbability;
    }

    @Override
    public ArrayList<Integer> makeCL() {
        ArrayList<Integer> _CL = new ArrayList<Integer>();
        for (int i = 0; i < ObjFunction.getDomainSize(); i++) {
            Integer cand = new Integer(i);
            _CL.add(cand);
        }
        return _CL;
    }

    @Override
    public ArrayList<Integer> makeRCL() {
        ArrayList<Integer> _RCL = new ArrayList<Integer>();
        return _RCL;
    }

    @Override
    public ArrayDeque<Integer> makeTL() {
        ArrayDeque<Integer> _TS = new ArrayDeque<Integer>(2 * tenure);
        for (int i = 0; i < 2 * tenure; i++) {
            _TS.add(fake);
        }
        return _TS;
    }

    @Override
    public void updateCL() {
        // do nothing
    }

    @Override
    public Solution<Integer> createEmptySol() {
        Solution<Integer> sol = new Solution<Integer>();
        sol.cost = 0.0;
        return sol;
    }

    /**
     * Neighborhood move adapted to support first-improving and best-improving.
     */
    @Override
    public Solution<Integer> neighborhoodMove() {

        Double bestDeltaCost = null;
        Integer bestCandIn = null, bestCandOut = null;

        updateCL();

        // Avaliar inserções
        for (Integer candIn : CL) {
            Double deltaCost = ObjFunction.evaluateInsertionCost(candIn, sol);
            boolean isTabu = TL.contains(candIn);
            boolean aspira = sol.cost + deltaCost < bestSol.cost;
            boolean moveAllowed = !isTabu || aspira;

            if (moveAllowed) {
                if (bestImproving) {
                    if (bestDeltaCost == null || deltaCost < bestDeltaCost) {
                        bestDeltaCost = deltaCost;
                        bestCandIn = candIn;
                        bestCandOut = null;
                    }
                } else { // first improving
                    if (deltaCost < 0) {
                        bestDeltaCost = deltaCost;
                        bestCandIn = candIn;
                        bestCandOut = null;
                        break; // para no primeiro movimento que melhora
                    }
                }
            }
        }

        if (bestImproving || bestCandIn == null) {
            // Avaliar remoções (somente se bestImproving ou ainda não achou movimento)
            for (Integer candOut : sol) {
                Double deltaCost = ObjFunction.evaluateRemovalCost(candOut, sol);
                boolean isTabu = TL.contains(candOut);
                boolean aspira = sol.cost + deltaCost < bestSol.cost;
                boolean moveAllowed = !isTabu || aspira;

                if (moveAllowed) {
                    if (bestImproving) {
                        if (bestDeltaCost == null || deltaCost < bestDeltaCost) {
                            bestDeltaCost = deltaCost;
                            bestCandIn = null;
                            bestCandOut = candOut;
                        }
                    } else {
                        if (deltaCost < 0) {
                            bestDeltaCost = deltaCost;
                            bestCandIn = null;
                            bestCandOut = candOut;
                            break;
                        }
                    }
                }
            }
        }

        if (bestImproving || bestCandIn == null && bestCandOut == null) {
            // Avaliar trocas (somente se bestImproving ou ainda não achou movimento)
            outerLoop:
            for (Integer candIn : CL) {
                for (Integer candOut : sol) {
                    Double deltaCost = ObjFunction.evaluateExchangeCost(candIn, candOut, sol);
                    boolean isTabu = TL.contains(candIn) || TL.contains(candOut);
                    boolean aspira = sol.cost + deltaCost < bestSol.cost;
                    boolean moveAllowed = !isTabu || aspira;

                    if (moveAllowed) {
                        if (bestImproving) {
                            if (bestDeltaCost == null || deltaCost < bestDeltaCost) {
                                bestDeltaCost = deltaCost;
                                bestCandIn = candIn;
                                bestCandOut = candOut;
                            }
                        } else {
                            if (deltaCost < 0) {
                                bestDeltaCost = deltaCost;
                                bestCandIn = candIn;
                                bestCandOut = candOut;
                                break outerLoop;
                            }
                        }
                    }
                }
            }
        }

        if (bestCandIn == null && bestCandOut == null) {
            // Nenhum movimento encontrado, aplicar estratégia tabu
            applyTabuStrategy();
            return null;
        }

        // Atualizar lista tabu
        TL.poll();
        if (bestCandOut != null) {
            sol.remove(bestCandOut);
            CL.add(bestCandOut);
            TL.add(bestCandOut);
        } else {
            TL.add(fake);
        }
        TL.poll();
        if (bestCandIn != null) {
            usageCount[bestCandIn]++;
            sol.add(bestCandIn);
            CL.remove(bestCandIn);
            TL.add(bestCandIn);
        } else {
            TL.add(fake);
        }

        ObjFunction.evaluate(sol);

        return null;
    }

    private void applyTabuStrategy() {
        switch (tabuStrategy) {
            case "intensificationRestart":
                // Reiniciar a busca com a melhor solução até agora
                sol.clear();
                sol.addAll(bestSol);
                sol.cost = bestSol.cost;
                break;
            case "diversificationRestart":
                // 1. Identificar variáveis raramente usadas
                // Encontra as 'k' variáveis com a menor contagem de uso
                int domainSize = ObjFunction.getDomainSize();
                int k = (int)(diversificationFixationRate * bestSol.size());
                List<Integer> sortedVariables = new ArrayList<>();

                for (int i = 0; i < domainSize; i++) {
                    if (bestSol.contains(i)) {
                        sortedVariables.add(i);
                    }
                }
                // Ordena as variáveis pela contagem de uso (do menor para o maior)
                sortedVariables.sort((a, b) -> Integer.compare(usageCount[a], usageCount[b]));

                // 2. Criar a nova solução forçando a inclusão das variáveis raras
                sol.clear();
                for (int i = 0; i < k && i < sortedVariables.size(); i++) {
                    sol.add(sortedVariables.get(i));
                }

                // 3. Completar a solução com um método aleatório
                for (int i = 0; i < domainSize; i++) {
                    if (!sol.contains(i) && Math.random() < diversificationSelectionProbability) {
                        sol.add(i);
                    }
                }

                ObjFunction.evaluate(sol);
                break;
            case "strategicOscillation":
                // Estratégia para alternar a solução (exemplo simples)
                if (!sol.isEmpty()) {
                    sol.remove(sol.iterator().next());
                }
                int randAdd = (int) (Math.random() * ObjFunction.getDomainSize());
                if (!sol.contains(randAdd)) {
                    sol.add(randAdd);
                }
                ObjFunction.evaluate(sol);
                break;
            default:
                // Estratégia padrão - não faz nada
                break;
        }
    }

    /**
     * Main para testar as configurações solicitadas.
     */
    public static void main(String[] args) throws IOException {

        int maxIter = 1000;
        int tenure1 = 7;
        int tenure2 = 15;
        String instance = "TS-Framework/TS-Framework/instances/qbf/qbf060";

        // Configuração 1 - padrão: first improving, tenure T1, estratégia default
        TS_QBF ts1 = new TS_QBF(tenure1, maxIter, instance, false, "default");
        long start1 = System.currentTimeMillis();
        Solution<Integer> best1 = ts1.solve();
        long end1 = System.currentTimeMillis();
        System.out.println("PADRÃO: " + best1 + " Tempo: " + (end1 - start1) / 1000.0 + " seg");

        // Configuração 2 - best improving, tenure T1, estratégia default
        TS_QBF ts2 = new TS_QBF(tenure1, maxIter, instance, true, "default");
        long start2 = System.currentTimeMillis();
        Solution<Integer> best2 = ts2.solve();
        long end2 = System.currentTimeMillis();
        System.out.println("PADRÃO+BEST: " + best2 + " Tempo: " + (end2 - start2) / 1000.0 + " seg");

        // Configuração 3 - first improving, tenure T2, estratégia default
        TS_QBF ts3 = new TS_QBF(tenure2, maxIter, instance, false, "default");
        long start3 = System.currentTimeMillis();
        Solution<Integer> best3 = ts3.solve();
        long end3 = System.currentTimeMillis();
        System.out.println("PADRÃO+TENURE: " + best3 + " Tempo: " + (end3 - start3) / 1000.0 + " seg");

        // Configuração 4 - first improving, tenure T1, estratégia diversificationRestart
        TS_QBF ts4 = new TS_QBF(tenure1, maxIter, instance, false, "diversificationRestart", 0.2, 0.5);
        long start4 = System.currentTimeMillis();
        Solution<Integer> best4 = ts4.solve();
        long end4 = System.currentTimeMillis();
        System.out.println("PADRÃO+DIVERSIFICATION: " + best4 + " Tempo: " + (end4 - start4) / 1000.0 + " seg");

        // Você pode adicionar mais configurações para as estratégias alternativas se quiser

    }
}
