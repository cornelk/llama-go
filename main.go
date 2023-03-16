package main

// #cgo CFLAGS:   -I. -O3 -DNDEBUG -std=c11 -fPIC -pthread -mavx -mavx2 -mfma -mf16c -msse3
// #cgo CXXFLAGS: -O3 -DNDEBUG -std=c++11 -fPIC -pthread -I.
// #include "main.h"
import "C"
import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
)

var (
	repeatLastN = 64
	seed        = -1
	threads     = 4
	tokens      = 128

	topK          = 40
	topP          = 0.95
	temp          = 0.80
	repeatPenalty = 1.30

	nCtx = 512 // context size

	options = map[string]interface{}{
		"repeat_last_n":  &repeatLastN, // last n tokens to penalize
		"repeat_penalty": &repeatPenalty,
		"seed":           &seed, // RNG seed, -1 will seed based on current time
		"temp":           &temp,
		"threads":        &threads,
		"tokens":         &tokens, // new tokens to predict
		"top_k":          &topK,
		"top_p":          &topP,
	}
)

func main() {
	var model string

	flags := flag.NewFlagSet(os.Args[0], flag.ExitOnError)
	flags.StringVar(&model, "m", "./models/7B/ggml-model-q4_0.bin", "path to q4_0.bin model file to load")
	flags.IntVar(&threads, "t", 4, "number of threads to use during computation")
	flags.IntVar(&tokens, "n", 128, "number of tokens to predict")

	err := flags.Parse(os.Args[1:])
	if err != nil {
		fmt.Printf("Parsing program arguments failed: %s", err)
		os.Exit(1)
	}

	state := C.llama_allocate_state()

	fmt.Printf("Loading model %s...\n", model)
	modelPath := C.CString(model)
	result := C.llama_bootstrap(modelPath, state, C.int(nCtx))
	if result != 0 {
		fmt.Println("Loading the model failed")
		os.Exit(1)
	}
	fmt.Printf("Model loaded successfully.\n")

	printSettings()
	reader := bufio.NewReader(os.Stdin)

	for {
		text := readMultiLineInput(reader)

		input := C.CString(text)
		params := C.llama_allocate_params(input, C.int(seed), C.int(threads), C.int(tokens), C.int(topK),
			C.float(topP), C.float(temp), C.float(repeatPenalty), C.int(repeatLastN))
		result = C.llama_predict(params, state)
		switch result {
		case 0:
		case 1:
			fmt.Println("\nPredicting failed")
			os.Exit(1)
		case 2:
			fmt.Printf(" <more token available>")
		}

		C.llama_free_params(params)

		fmt.Printf("\n\n")
	}
}

// readMultiLineInput reads input until an empty line is entered.
func readMultiLineInput(reader *bufio.Reader) string {
	var lines []string
	fmt.Print(">>> ")

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				os.Exit(0)
			}
			fmt.Printf("Reading the prompt failed: %s", err)
			os.Exit(1)
		}

		if len(strings.TrimSpace(line)) == 0 {
			break
		}

		optionChanged, err := handleParameterChange(line)
		if err != nil {
			fmt.Printf("Reading the prompt failed: %s", err)
			os.Exit(1)
		}
		if optionChanged {
			lines = nil
			fmt.Print(">>> ")
			continue
		}

		lines = append(lines, line)
	}

	text := strings.Join(lines, "")
	return text
}

// handleParameterChange parses the input for any parameter changes.
// This is a generic function that can handle int and float type parameters.
// The parameters need to be referenced by pointer in the options map.
func handleParameterChange(input string) (bool, error) {
	optionChanged := false
	words := strings.Split(input, " ")

	for _, word := range words {
		parsed := strings.Split(word, "=")

		if len(parsed) < 2 {
			break
		}

		s := strings.TrimSpace(parsed[0])
		opt, ok := options[s]
		if !ok {
			break
		}

		val := reflect.ValueOf(opt)
		if val.Kind() != reflect.Ptr {
			return false, fmt.Errorf("option %s is not a pointer", s)
		}
		val = val.Elem()
		argument := strings.TrimSpace(parsed[1])
		optionChanged = true

		switch val.Kind() {
		case reflect.Int:
			i, err := strconv.ParseInt(argument, 10, 64)
			if err != nil {
				return false, fmt.Errorf("parsing value '%s' as int: %w", argument, err)
			}
			val.SetInt(i)

		case reflect.Float32, reflect.Float64:
			f, err := strconv.ParseFloat(argument, 64)
			if err != nil {
				return false, fmt.Errorf("parsing value '%s' as float: %w", argument, err)
			}
			val.SetFloat(f)

		default:
			return false, fmt.Errorf("unsupported option %s type %T", s, opt)
		}
	}

	if optionChanged {
		printSettings()
	}
	return optionChanged, nil
}

// printSettings outputs the current settings, alphabetically sorted.
func printSettings() {
	var settings sort.StringSlice
	for setting, value := range options {
		val := reflect.ValueOf(value)
		if val.Kind() == reflect.Ptr {
			val = val.Elem()
		}
		settings = append(settings, fmt.Sprintf("%s=%v", setting, val.Interface()))
	}
	sort.Sort(settings)
	s := strings.Join(settings, " ")
	fmt.Printf("Settings: %s\n\n", s)
}
