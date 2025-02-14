# Use Python 3.13 RC slim image as base
FROM python:3.13-rc-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (including autotools-dev and libglib2.0-0)
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    autotools-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Build & install TA-Lib from source, copying modern config.guess/config.sub
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && cp /usr/share/misc/config.guess . \
    && cp /usr/share/misc/config.sub . \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

# Verify TA-Lib installation
RUN ls -la /usr/include/ta-lib || echo "TA-Lib headers not found in /usr/include/ta-lib" \
    && ls -la /usr/lib/libta_lib* || echo "TA-Lib library not found in /usr/lib"

# Create symlink and update linker cache
RUN if [ -f /usr/lib/libta_lib.so ]; then \
      ln -s /usr/lib/libta_lib.so /usr/lib/libta-lib.so; \
    fi \
    && ldconfig

# Set TA-Lib environment variables for Python wrapper
ENV TA_LIBRARY_PATH=/usr/lib \
    TA_INCLUDE_PATH=/usr/include

# Verify environment variables
RUN echo "TA_LIBRARY_PATH=$TA_LIBRARY_PATH" \
    && echo "TA_INCLUDE_PATH=$TA_INCLUDE_PATH"

# Copy only requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies with verbose output for debugging
RUN pip install --no-cache-dir -v ta-lib && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Command to run your Python script
CMD ["python", "trade_server.py"] 