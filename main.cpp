#include <iostream>
#include <future>
#include <vector>
#include <cstring>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <cerrno>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iterator>
#include <iomanip>
#include <gmpxx.h>
#include <gmp.h>

struct Options
{
    unsigned threads = 1;
    unsigned granularity = 1;
    unsigned terms = 1'000'000;
    const char* outputFile = nullptr;
    bool verbose = true;
};

unsigned long strToLong(const char* str, int base = 10)
{
    if(!str) throw std::invalid_argument("nullptr argument");
    char *end;
    errno = 0;
    unsigned long res = std::strtoul(str, &end, base);
    if(errno == ERANGE || *end)
        throw std::invalid_argument("could not convert to number");
    return res;
}

Options parseArgs(int argc, char** argv)
{
    Options args;
    while(argc)
    {
        using namespace std::string_literals;
        if(!strcmp(*argv, "-q"))
        {
            args.verbose = false;
            argc--; argv++;
            continue;
        }
        else if(!strcmp(*argv, "-p"))
            args.terms = strToLong(argv[1], 10);
        else if(!strcmp(*argv, "-t"))
            args.threads = strToLong(argv[1], 10);
        else if(!strcmp(*argv, "-g"))
            args.granularity = strToLong(argv[1], 10);
        else if(!strcmp(*argv, "-o"))
        {
            if(argc < 2) throw std::invalid_argument("argument "s + *argv + " expects filename");
            args.outputFile = argv[1];
        }
        else throw std::invalid_argument("argument "s + *argv + " not recognized");
        argc -= 2; argv += 2;
    }
    if(!args.granularity || !args.terms || !args.threads)
        throw std::invalid_argument("invalid argument given");
    return args;
}

template<class Time>
std::ostream& printTimes(std::ostream& os, const std::vector<Time>& times)
{
    for(std::size_t i=0; i<times.size(); i++)
            os << " thread #" << std::setw(2) << i+1 << ": " << times[i].count() << " ms\n";
    return os;
}

struct BigFraction
{
    mpz_class num, den;
    unsigned long id;
    BigFraction& operator++()
    {
        num += den;
        return *this;
    }
};

BigFraction& Combine(BigFraction& a, const BigFraction& b)
{
    a.num += b.num*a.den;
    a.den *= b.den;
    return a;
}

BigFraction partiallyCompute(unsigned long start, unsigned long end, unsigned long id)
{
    if(start == end) return {1, start, id};
    if(end-start == 1)
    {
        BigFraction res{0, start, id};
        res.num = end+1;
        res.den *= end;
        return res;
    }
    unsigned long m = ((unsigned long long)start+end)/2;
    BigFraction tmp = partiallyCompute(m+1, end, id);
    Combine(tmp, partiallyCompute(start, m, id));
    return tmp;
}

std::vector<BigFraction> partialSums(const Options& o, std::vector<std::chrono::milliseconds>& times)
{
    unsigned long lastTermIndex = o.terms-1;
    unsigned long start = 1, step = lastTermIndex/o.granularity;
    std::vector<std::future<BigFraction>> futures(o.threads);
    std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> startTimes(o.threads);
    std::vector<BigFraction> res((lastTermIndex+step)/(step+1));
    unsigned long id = res.size() - 1;
    for(std::size_t i=0; i<futures.size(); i++)
        if(start <= lastTermIndex)
        {
            startTimes[i] = std::chrono::high_resolution_clock::now();
            futures[i] = std::async(std::launch::async, partiallyCompute, start, std::min(start+step, lastTermIndex), id--);
            start += step+1;
        }
        else break;
    while(start <= lastTermIndex)
    {
        for(std::size_t i=0; i<futures.size(); i++)
            if(futures[i].valid())
            {
                BigFraction tmp = futures[i].get();
                if(start <= lastTermIndex)
                {
                    futures[i] = std::async(std::launch::async, partiallyCompute, start, std::min(start+step, lastTermIndex), id--);
                    start += step+1;
                }
                else
                    times[i] += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-startTimes[i]);
                res[tmp.id] = std::move(tmp);
            }
            else break;
    }
    for(std::size_t i=0; i<futures.size(); i++)
        if(futures[i].valid())
        {
            BigFraction tmp = futures[i].get();
            times[i] += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-startTimes[i]);
            res[tmp.id] = std::move(tmp);
        }
        else break;
    return res;
}

template<class Iter>
BigFraction& addFractions(Iter first, Iter last, unsigned long id)
{
    if(last-first == 1)
    {
        first->id = id;
        return *first;
    }
    auto m = first+(last-first)/2;
    return Combine(addFractions(first, m, id), addFractions(m, last, id));
}

BigFraction addSums(std::vector<BigFraction>& ps, const Options& o, std::vector<std::chrono::milliseconds>& times)
{
    if(ps.empty()) return {1, 1};
    std::vector<BigFraction> res;
    std::vector<std::future<BigFraction&>> futures(o.threads);
    std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> startTimes(o.threads);
    long step = std::max(o.granularity/(4*o.threads), 2u)/*16*/;
    while(ps.size() > 1)
    {
        res.resize((ps.size()+step-1)/step);
        unsigned long id = 0;
        auto start = ps.begin();
        for(std::size_t i=0; i<futures.size(); i++)
            if(start < ps.end())
            {
                auto end = ps.end()-start >= step? start+step: ps.end();
                startTimes[i] = std::chrono::high_resolution_clock::now();
                futures[i] = std::async(std::launch::async, addFractions<std::vector<BigFraction>::iterator>, start, end, id++);
                start = end;
            }
            else break;
        while(start < ps.end())
        {
            for(std::size_t i=0; i<futures.size(); i++)
                if(futures[i].valid())
                {
                    BigFraction& tmp = futures[i].get();
                    if(start < ps.end())
                    {
                        auto end = ps.end()-start >= step? start+step: ps.end();
                        futures[i] = std::async(std::launch::async, addFractions<std::vector<BigFraction>::iterator>, start, end, id++);
                        start = end;
                    }
                    else
                        times[i] += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-startTimes[i]);
                    res[tmp.id] = std::move(tmp);
                }
                else break;
        }
        for(std::size_t i=0; i<futures.size(); i++)
            if(futures[i].valid())
            {
                BigFraction& tmp = futures[i].get();
                res[tmp.id] = std::move(tmp);
                times[i] += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-startTimes[i]);
            }
            else break;
        step = 2;
        ps = std::move(res);
    }
    return std::move(++ps.back());
}

template<class BigNumber>
std::ostream& print(std::ostream& os, const BigNumber& n, unsigned long precision)
{
    os.precision(precision);
    return os << std::fixed << n << '\n';
}

int main(int argc, char** argv) try
{
    Options args = parseArgs(argc-1, argv+1);
    constexpr unsigned minTerms = 20;
    if(args.terms < minTerms)
    {
        std::cerr << "computed terms must be at least " << minTerms << '\n';
        return 1;
    }
    if(args.terms < 10000 && args.threads > 1)
    {
        args.granularity = args.threads = 1;
        if(args.verbose) std::cout << "too little work... changing to 1 thread mode" << std::endl;
    }
    std::vector<std::chrono::milliseconds> times(args.threads);
    if(args.verbose) std::cout << "starting computing of partial sums... " << std::flush;
    auto start = std::chrono::high_resolution_clock::now(), start2 = start;
    auto ps = partialSums(args, times);
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start2).count() << " ms elapsed" << std::endl;
    if(args.verbose)
    {
        std::cout << "Times after partialSums():\n";
        printTimes(std::cout, times);
    }
    if(args.verbose) std::cout << "starting addition of partial sums... " << std::flush;
    start2 = std::chrono::high_resolution_clock::now();
    BigFraction final = addSums(ps, args, times);
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start2).count() << " ms elapsed" << std::endl;
    std::size_t digits = mpz_sizeinbase(final.den.get_mpz_t(), 10)-10;
    if(args.verbose) std::cout << "starting division... " << std::flush;
    start2 = std::chrono::high_resolution_clock::now();
    mpf_class result = mpf_class(final.num, 4*digits)/final.den;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start2).count() << " ms elapsed" << std::endl;
    if(args.verbose) std::cout << "starting printing... " << std::flush;
    start2 = std::chrono::high_resolution_clock::now();
    if(args.outputFile)
    {
        std::ofstream of(args.outputFile);
        print(of, result, digits);
    }
    else print(std::cout, result, digits);
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start2).count() << " ms elapsed" << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    if(args.verbose)
    {
        std::cout << "Final times:\n";
        printTimes(std::cout, times);
        std::cout << std::setw(13) << "main thread: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << " ms\n";
        std::cout << digits << " digits of 'e' were computed" << std::endl;
    }
}
catch(const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return 1;
}
