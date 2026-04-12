import CoreML
import Foundation

@main struct App {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            fputs("Usage: ane-costplan [-j report.json] <model.mlmodelc>\n", stderr); exit(1)
        }
        
        var jsonPath: String? = nil
        let modelPath = args[args.count - 1]
        if args.count >= 4 && args[1] == "-j" { jsonPath = args[2] }
        
        let url = URL(fileURLWithPath: modelPath)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        
        do {
            let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
            guard case .program(let prog) = plan.modelStructure else {
                fputs("Not an mlProgram\n", stderr); exit(1)
            }
            
            var entries: [(idx: Int, type: String, cost: Double)] = []
            var totalCost = 0.0
            var aneCnt = 0, cpuCnt = 0, otherCnt = 0
            
            for (_, function) in prog.functions {
                for (i, op) in function.block.operations.enumerated() {
                    let opType = op.operatorName
                    if opType.hasPrefix("const") { continue }
                    
                    let cost = plan.estimatedCost(of: op)?.weight ?? 0.0
                    totalCost += cost
                    
                    let hasUsage = plan.deviceUsage(for: op) != nil
                    if hasUsage { aneCnt += 1 } else { otherCnt += 1 }
                    
                    entries.append((idx: i, type: opType, cost: cost))
                }
            }
            
            // Sort descending by cost
            entries.sort { $0.cost > $1.cost }
            
            // Print using simple string concatenation (avoid format crashes)
            print("  idx  op_type                           cost         pct")
            print(String(repeating: "─", count: 65))
            for e in entries where e.cost > 0 {
                let pct = totalCost > 0 ? e.cost / totalCost * 100 : 0
                let line = "  \(String(e.idx).padding(toLength: 4, withPad: " ", startingAt: 0))  " +
                           "\(e.type.padding(toLength: 30, withPad: " ", startingAt: 0))  " +
                           "\(String(format: "%.6f", e.cost).padding(toLength: 12, withPad: " ", startingAt: 0))  " +
                           "\(String(format: "%4.1f", pct))%"
                print(line)
            }
            print(String(repeating: "─", count: 65))
            print("Total cost: \(String(format: "%.6f", totalCost))  |  ops: \(entries.count)  |  with_device: \(aneCnt)  no_device: \(otherCnt)")
            
            // Top 10
            print("\n══ Top 10 by cost ══")
            for (rank, e) in entries.prefix(10).enumerated() {
                let pct = totalCost > 0 ? e.cost / totalCost * 100 : 0
                print("  \(rank+1). [\(e.idx)] \(e.type) → \(String(format: "%.6f", e.cost)) (\(String(format: "%.1f", pct))%)")
            }
            
            // JSON
            if let jp = jsonPath {
                var jsonOps: [[String: Any]] = []
                for e in entries {
                    jsonOps.append(["index": e.idx, "type": e.type, "cost": e.cost])
                }
                let report: [String: Any] = [
                    "model": modelPath, "total_cost": totalCost,
                    "total_ops": entries.count, "with_device": aneCnt, "no_device": otherCnt,
                    "operations": jsonOps,
                ]
                let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                try data.write(to: URL(fileURLWithPath: jp))
                print("\nJSON → \(jp)")
            }
        } catch {
            fputs("Error: \(error)\n", stderr); exit(1)
        }
    }
}
