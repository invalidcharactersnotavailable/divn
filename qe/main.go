package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"
)

type Equation struct {
	Query      string `json:"query"`
	Completion string `json:"completion"`
}

func compute(a int, b int, op string) (int, bool) {
	switch op {
	case "+":
		return a + b, true
	case "-":
		return a - b, true
	case "*":
		return a * b, true
	case "/":
		if b == 0 || a%b != 0 {
			return 0, false
		}
		return a / b, true
	}
	return 0, false
}

func generateEquation(op string) Equation {
	ops := []string{"+", "-", "*", "/"}
	for {
		a := rand.Intn(100)
		b := rand.Intn(100)

		actualOp := op
		if op == "all" {
			actualOp = ops[rand.Intn(len(ops))]
		}

		result, valid := compute(a, b, actualOp)
		if valid {
			query := fmt.Sprintf("%d%s%d", a, actualOp, b)
			completion := fmt.Sprintf("%s=%d", query, result)
			return Equation{Query: query, Completion: completion}
		}
	}
}

func main() {
	count := flag.Int("n", 1, "number of equations")
	op := flag.String("op", "all", "operation type: +, -, *, /, all")
	outFile := flag.String("out", "output.json", "output file name")
	flag.Parse()

	validOps := map[string]bool{"+": true, "-": true, "*": true, "/": true, "all": true}
	if !validOps[*op] {
		fmt.Println("error: invalid operation type. use +, -, *, /, or all")
		os.Exit(1)
	}

	rand.Seed(time.Now().UnixNano())
	var equations []Equation
	for i := 0; i < *count; i++ {
		equations = append(equations, generateEquation(*op))
	}

	f, err := os.Create(*outFile)
	if err != nil {
		fmt.Println("error creating file:", err)
		os.Exit(1)
	}
	defer f.Close()

	encoder := json.NewEncoder(f)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(equations); err != nil {
		fmt.Println("error writing json:", err)
		os.Exit(1)
	}
}
